"""Map-aware receiver: use the jamming map to protect decoding.

The standard receiver trusts every pilot and every data symbol. With a jamming map the gNB can
treat the flagged cells differently. Two ways are implemented (``mode``):

**hard** (as in the classic literature, e.g. Li et al., MILCOM 2012):

1. *Repair the channel estimate.* Pilots in PRBs whose DMRS is flagged are dropped, and the
   channel on every subcarrier is interpolated from the clean pilots with an LMMSE filter. The
   filter uses the same channel model as the DMRS residual feature (delays spread uniformly up to
   ``tau_max``), so one model both detects the attack and repairs it.
2. *Erase jammed data.* The soft bits (LLRs) of data symbols in flagged cells are set to 0, so
   the LDPC decoder treats them as unknown instead of trusting wrong values.

**soft**: the map says *where*, the receiver estimates *how strong* from the flagged cells:

1. Pilots of a flagged PRB are kept, but the LMMSE filter is told their noise is larger: the
   PRB's DMRS residual ``R`` (about 1 without jamming, ``1 + JNR`` under jamming) multiplies their
   noise variance. Strongly jammed pilots are effectively ignored; a false alarm (``R`` about 1)
   costs nothing.
2. The jamming power on the flagged data cells of each PRB is estimated from their excess
   received power (received energy minus the expected signal and noise energy), and added to the
   noise variance of those symbols before demapping. The LLRs shrink by exactly the right amount:
   to almost 0 under strong jamming, hardly at all after a false alarm.

**dmrs** (no map, what a receiver can do with its DMRS alone): a PRB is declared interfered if its
DMRS residual exceeds the analytic 1% threshold (``jamloc.theory``); its pilots are down-weighted
as in *soft*, and the interference estimated on the DMRS is assumed to last the whole slot, so all
data symbols of the PRB get the extra noise variance. This is blind to data-only jamming, and it
wrongly penalizes the clean data of a pilot-jammed PRB.

Two more references without a map separate what the map adds from what the rest of the receiver
adds:

**lmmse**: the same LMMSE interpolation from all pilots, with the noise variance of the standard
receiver, and plain demapping. It differs from the standard receiver (Sionna's LS estimate with
linear interpolation) only in the interpolator.

**slot**: an interference-aware receiver that measures noise plus interference once per slot, as
receivers without any localization do: every pilot's noise variance is multiplied by the mean
DMRS residual of the allocation, and the mean excess energy over all data symbols of the slot is
added to the noise variance of every data symbol.

Why LMMSE and not a least-squares fit: dropping a block of pilots leaves a gap, and a plain fit
across the gap amplifies the noise enormously. LMMSE also gives an error variance for every
subcarrier. Where the gap is wider than the channel's coherence bandwidth, that variance
approaches the channel power, the equalizer reports a large effective noise, and the LLRs there
shrink towards zero by themselves: repair turns smoothly into erasure.

If the hard map flags every DMRS cell there is no clean pilot to interpolate from. The receiver
then uses Sionna's standard channel estimate, exactly like the unprotected receiver, and only
erases the flagged data cells.
"""

import math

import numpy as np
import torch
from sionna.phy import Block
from sionna.phy.mapping import Demapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.nr import LayerDemapper, PUSCHLSChannelEstimator, PUSCHTransmitter, TBDecoder
from sionna.phy.ofdm import LMMSEEqualizer

from jamloc import theory
from jamloc.config import SimConfig
from jamloc.features import dmrs_ls, residual_per_prb
from jamloc.receiver import data_positions


def lmmse_interpolation(
    h_ls: torch.Tensor,
    pilot_sc: torch.Tensor,
    clean: torch.Tensor,
    s2: torch.Tensor,
    num_sc: int,
    scs: float,
    tau_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LMMSE channel estimate on all ``num_sc`` subcarriers from the clean pilots only.

    ``h_ls`` ``[B, N_p]``: LS estimates at the pilots (subcarriers ``pilot_sc``). ``clean``
    ``[B, N_p]``: which pilots to use. ``s2``: LS noise variance, ``[B]`` (the same on every
    pilot) or ``[B, N_p]`` (per pilot).
    Returns ``h_hat`` ``[B, num_sc]`` and its error variance ``[B, num_sc]`` (channel power 1).
    """
    # Everything below runs on the CPU and the results go back to the input's device. On CUDA,
    # PyTorch hands batched complex double-precision solves to the MAGMA library, which crashes
    # with an illegal memory access on the CUDA 13 build Colab uses (reproduced with random
    # matrices, no project code involved). The matrices are small (one 72 x 72 per slot for a
    # 12-PRB allocation), so the CPU solves them in about a millisecond, on every machine alike.
    device = h_ls.device
    h_ls, pilot_sc, clean, s2 = h_ls.cpu(), pilot_sc.cpu(), clean.cpu(), s2.cpu()

    # Frequency correlation of a channel with delays spread uniformly over [0, tau_max].
    # Computed in double precision: the matrix is close to singular (that is the point of the
    # model), and single precision would lose the small eigenvalues.
    k = torch.arange(num_sc, dtype=torch.float64)
    x = (k[:, None] - k[None, :]) * scs * tau_max
    corr = torch.exp(-1j * math.pi * x) * torch.sinc(x)  # [num_sc, num_sc]
    r_ap = corr[:, pilot_sc]  # all subcarriers x pilots
    r_pp = corr[pilot_sc][:, pilot_sc]  # pilots x pilots

    # Use only the clean pilots. To keep the whole batch in one linear solve, an unclean pilot is
    # decoupled instead of removed: its row and column of the pilot correlation are zeroed, its
    # diagonal is 1 and its observation is 0, so it contributes nothing to the estimate.
    c = clean.to(torch.complex128)
    s2 = s2.to(torch.float64)
    s2 = s2[:, None].expand(clean.shape) if s2.dim() == 1 else s2
    diag = torch.where(clean, s2, torch.ones_like(s2))
    a = r_pp[None] * (c[:, :, None] * c[:, None, :]) + torch.diag_embed(diag.to(torch.complex128))
    g = r_ap[None] * c[:, None, :]  # [B, num_sc, N_p]

    # h_hat = R_ac (R_cc + S)^-1 h_c, S = diag(noise variance of each pilot)
    observed = torch.where(clean, h_ls.to(torch.complex128), 0)  # also drops NaNs of unused pilots
    weights = torch.linalg.solve(a, observed[..., None])
    h_hat = (g @ weights)[..., 0]

    # err_k = R_kk - R_kc (R_cc + s2 I)^-1 R_ck, with R_kk = 1
    z = torch.linalg.solve(a, g.conj().transpose(1, 2))  # [B, N_p, num_sc]
    err_var = 1 - (g * z.transpose(1, 2)).sum(-1).real
    h_hat = h_hat.to(device=device, dtype=h_ls.dtype)
    return h_hat, err_var.clamp(0, 1).to(device=device, dtype=torch.float32)


MODES = ("hard", "soft", "dmrs", "lmmse", "slot")


class MapAwareReceiver(Block):
    """Receiver for one allocation size that uses a jamming map ``cells`` ``[B, n, 14]`` (bool).

    Same chain and outputs as ``PUSCHReceiverChain``; only the channel estimate and the LLRs differ.
    """

    def __init__(self, tx: PUSCHTransmitter, cfg: SimConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tx = tx
        self.cfg = cfg or SimConfig()
        rg = tx.resource_grid
        sm = StreamManagement(np.ones([1, 1], bool), 1)
        self.standard_estimator = PUSCHLSChannelEstimator(
            rg,
            tx._dmrs_length,
            tx._dmrs_additional_position,
            tx._num_cdm_groups_without_data,
            interpolation_type="lin",
        )
        self.equalizer = LMMSEEqualizer(rg, sm)
        self.demapper = Demapper("maxlog", "qam", tx._num_bits_per_symbol)
        self.layer_demapper = LayerDemapper(
            tx._layer_mapper, num_bits_per_symbol=tx._num_bits_per_symbol
        )
        self.decoder = TBDecoder(tx._tb_encoder)
        self.data_sym, self.data_sc = data_positions(tx)

    def call(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
        cells: torch.Tensor | None,
        decode: bool = True,
        mode: str = "hard",
    ):
        """``mode``: ``"hard"``, ``"soft"`` (both use ``cells``), or ``"dmrs"``, ``"lmmse"``,
        ``"slot"`` (no map, ``cells`` is ignored)."""
        cfg = self.cfg
        num_sc = y.shape[-1]
        num_prb = num_sc // 12
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode in ("hard", "soft"):
            cells = cells.to(device=y.device, dtype=torch.bool)
            if tuple(cells.shape[1:]) != (num_prb, cfg.num_symbols):
                raise ValueError(
                    f"cells must cover the allocation, [B, {num_prb}, {cfg.num_symbols}]; got "
                    f"{tuple(cells.shape)} (cut a carrier map to the scheduled PRBs first)"
                )

        # 1. LS estimates at the DMRS pilots and the residual of every PRB
        h_ls, pilot_sc, pilot_power = dmrs_ls(y, self.tx)
        pilot_prb = pilot_sc // 12
        s2 = no / pilot_power  # LS noise variance per pilot without jamming
        r, interfered = None, None
        if mode == "hard":
            clean = ~cells[:, pilot_prb, cfg.dmrs_symbol]  # pilots of unflagged PRBs only
            pilot_var = s2
        elif mode == "lmmse":
            clean = torch.ones_like(h_ls, dtype=torch.bool)  # every pilot, equally trusted
            pilot_var = s2
        else:
            r = residual_per_prb(h_ls, pilot_sc, no, pilot_power, cfg).clamp_min(1)  # [B, n]
            if mode == "slot":  # one level for the whole allocation: the mean residual
                r = r.mean(dim=1, keepdim=True).expand_as(r)
                interfered = torch.ones_like(r, dtype=torch.bool)
            elif mode == "dmrs":  # interfered PRBs: residual above the analytic 1% threshold
                thr = torch.tensor(
                    [theory.threshold(num_prb, b, 0.01, cfg) for b in range(num_prb)],
                    device=r.device,
                    dtype=r.dtype,
                )
                interfered = r > thr
            else:
                interfered = cells[:, :, cfg.dmrs_symbol]
            inflate = torch.where(interfered, r, torch.ones_like(r))  # [B, n]
            clean = torch.ones_like(h_ls, dtype=torch.bool)  # every pilot, weighted
            pilot_var = s2[:, None] * inflate[:, pilot_prb]

        # 2. Repair: LMMSE interpolation, the same on all 14 symbols
        h_sc, err_sc = lmmse_interpolation(
            h_ls, pilot_sc, clean, pilot_var, num_sc, cfg.scs, cfg.tau_max
        )
        shape = (y.shape[0], 1, 1, 1, 1, cfg.num_symbols, num_sc)  # Sionna's h_hat layout
        h_hat = h_sc[:, None, None, None, None, None, :].expand(shape)
        err_var = err_sc[:, None, None, None, None, None, :].expand(shape)

        # 3. No clean pilot at all: nothing to interpolate from, use the standard estimate
        no_clean = ~clean.any(dim=1)
        if no_clean.any():
            h_std, err_std = self.standard_estimator(y, no)
            pick = no_clean[:, None, None, None, None, None, None]
            h_hat = torch.where(pick, h_std, h_hat)
            err_var = torch.where(pick, err_std.expand(shape), err_var)

        # 4. Equalize, as in the standard receiver
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)

        # 5. Protect the data symbols, then demap (two bits per QPSK symbol)
        data_prb = self.data_sc // 12
        bits = self.tx._num_bits_per_symbol
        if mode == "hard":
            llr = self.demapper(x_hat, no_eff)
            flagged = cells[:, data_prb, self.data_sym].repeat_interleave(bits, dim=1)
            llr = llr.masked_fill(flagged[:, None, None, :], 0.0)
        elif mode == "lmmse":
            llr = self.demapper(x_hat, no_eff)
        else:
            jam_var = self._jamming_variance(y, no, h_sc, err_sc, cells, r, interfered, mode)
            # For one antenna the unbiased LMMSE output is y / h_hat and its effective noise is
            # (err_var + no) / |h_hat|^2; extra noise of variance v adds v / |h_hat|^2
            h2 = (h_hat[:, 0, 0, 0, 0, self.data_sym, self.data_sc].abs() ** 2).clamp_min(1e-12)
            no_eff = no_eff + (jam_var / h2)[:, None, None, :]
            llr = self.demapper(x_hat, no_eff)

        out = {
            "h_hat": h_hat,
            "err_var": err_var,
            "x_hat": x_hat[:, 0, 0],
            "no_eff": no_eff[:, 0, 0],
            "llr": llr,
        }
        if decode:
            b_hat, tb_crc = self.decoder(self.layer_demapper(llr))
            out["b_hat"], out["tb_crc"] = b_hat, tb_crc
        return out

    def _jamming_variance(self, y, no, h_sc, err_sc, cells, r, interfered, mode):
        """Estimated jamming power ``[B, num_data]`` on every data symbol (0 where none)."""
        data_prb = self.data_sc // 12
        if mode == "slot":  # mean excess energy over all data symbols, on every data symbol
            excess = self._excess_energy(y, no, h_sc, err_sc)
            return excess.mean(dim=1, keepdim=True).clamp_min(0).expand_as(excess)
        if mode == "dmrs":
            # Interference measured on the DMRS, assumed to last the whole slot:
            # residual R = 1 + JNR, so the jamming power is N0 (R - 1)
            per_prb = torch.where(interfered, no[:, None] * (r - 1), torch.zeros_like(r))
            return per_prb[:, data_prb]

        excess = self._excess_energy(y, no, h_sc, err_sc)
        flagged = cells[:, data_prb, self.data_sym]  # [B, num_data]
        # Average over the flagged data symbols of each PRB (about 156 symbols per jammed PRB)
        num_prb = r.shape[1]
        total = excess.new_zeros(excess.shape[0], num_prb).index_add_(
            1, data_prb, torch.where(flagged, excess, torch.zeros_like(excess))
        )
        count = excess.new_zeros(excess.shape[0], num_prb).index_add_(
            1, data_prb, flagged.to(excess.dtype)
        )
        per_prb = (total / count.clamp_min(1)).clamp_min(0)
        return torch.where(flagged, per_prb[:, data_prb], torch.zeros_like(excess))

    def _excess_energy(self, y, no, h_sc, err_sc):
        """``[B, num_data]``: received energy of each data symbol minus what the UE's signal (unit
        energy, seen through the estimated channel and its error) and the noise explain."""
        y_data = y[:, 0, 0, self.data_sym, self.data_sc]
        return (
            y_data.abs() ** 2
            - h_sc[:, self.data_sc].abs() ** 2
            - err_sc[:, self.data_sc]
            - no[:, None]
        )
