"""Map-aware receiver: use the jamming map to protect decoding.

The standard receiver trusts every pilot and every data symbol. With a jamming map the gNB can do
two things instead:

1. **Repair the channel estimate.** Pilots in PRBs whose DMRS is flagged are dropped, and the
   channel on every subcarrier is interpolated from the clean pilots with an LMMSE filter. The
   filter uses the same channel model as the DMRS residual feature (delays spread uniformly up to
   ``tau_max``), so one model both detects the attack and repairs it.
2. **Erase jammed data.** The soft bits (LLRs) of data symbols in flagged cells are set to 0, so
   the LDPC decoder treats them as unknown instead of trusting wrong values.

Why LMMSE and not a least-squares fit: dropping a block of pilots leaves a gap, and a plain fit
across the gap amplifies the noise enormously. LMMSE also gives an error variance for every
subcarrier. Where the gap is wider than the channel's coherence bandwidth, that variance
approaches the channel power, the equalizer reports a large effective noise, and the LLRs there
shrink towards zero by themselves: repair turns smoothly into erasure.

If the map flags every DMRS cell there is no clean pilot to interpolate from. The receiver then
uses Sionna's standard estimate, exactly like the unprotected receiver, and only erases flagged
data cells; it can never do worse than no protection.
"""

import math

import numpy as np
import torch
from sionna.phy import Block
from sionna.phy.mapping import Demapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.nr import LayerDemapper, PUSCHLSChannelEstimator, PUSCHTransmitter, TBDecoder
from sionna.phy.ofdm import LMMSEEqualizer

from jamloc.config import SimConfig
from jamloc.features import dmrs_ls
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
    ``[B, N_p]``: which pilots to use. ``s2`` ``[B]``: LS noise variance per pilot.
    Returns ``h_hat`` ``[B, num_sc]`` and its error variance ``[B, num_sc]`` (channel power 1).
    """
    # Frequency correlation of a channel with delays spread uniformly over [0, tau_max].
    # Computed in double precision: the matrix is close to singular (that is the point of the
    # model), and single precision would lose the small eigenvalues.
    device = h_ls.device
    k = torch.arange(num_sc, device=device, dtype=torch.float64)
    x = (k[:, None] - k[None, :]) * scs * tau_max
    corr = torch.exp(-1j * math.pi * x) * torch.sinc(x)  # [num_sc, num_sc]
    r_ap = corr[:, pilot_sc]  # all subcarriers x pilots
    r_pp = corr[pilot_sc][:, pilot_sc]  # pilots x pilots

    # Use only the clean pilots. To keep the whole batch in one linear solve, an unclean pilot is
    # decoupled instead of removed: its row and column of the pilot correlation are zeroed, its
    # diagonal is 1 and its observation is 0, so it contributes nothing to the estimate.
    c = clean.to(torch.complex128)
    s2 = s2.to(torch.float64)
    diag = torch.where(clean, s2[:, None], torch.ones_like(s2)[:, None])
    a = r_pp[None] * (c[:, :, None] * c[:, None, :]) + torch.diag_embed(diag.to(torch.complex128))
    g = r_ap[None] * c[:, None, :]  # [B, num_sc, N_p]

    # h_hat = R_ac (R_cc + s2 I)^-1 h_c
    weights = torch.linalg.solve(a, (h_ls.to(torch.complex128) * c)[..., None])
    h_hat = (g @ weights)[..., 0]

    # err_k = R_kk - R_kc (R_cc + s2 I)^-1 R_ck, with R_kk = 1
    z = torch.linalg.solve(a, g.conj().transpose(1, 2))  # [B, N_p, num_sc]
    err_var = 1 - (g * z.transpose(1, 2)).sum(-1).real
    return h_hat.to(h_ls.dtype), err_var.clamp(0, 1).to(torch.float32)


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

    def call(self, y: torch.Tensor, no: torch.Tensor, cells: torch.Tensor, decode: bool = True):
        cfg = self.cfg
        cells = cells.to(device=y.device, dtype=torch.bool)
        num_sc = y.shape[-1]
        if tuple(cells.shape[1:]) != (num_sc // 12, cfg.num_symbols):
            raise ValueError(
                f"cells must cover the allocation, [B, {num_sc // 12}, {cfg.num_symbols}]; got "
                f"{tuple(cells.shape)} (cut a carrier map to the scheduled PRBs first)"
            )

        # 1. LS estimates at the DMRS pilots; a pilot is clean if its PRB's DMRS cell is not flagged
        h_ls, pilot_sc, pilot_power = dmrs_ls(y, self.tx)
        clean = ~cells[:, pilot_sc // 12, cfg.dmrs_symbol]

        # 2. Repair: LMMSE interpolation from the clean pilots, the same on all 14 symbols
        h_sc, err_sc = lmmse_interpolation(
            h_ls, pilot_sc, clean, no / pilot_power, num_sc, cfg.scs, cfg.tau_max
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

        # 4. Equalize and demap, as in the standard receiver
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)

        # 5. Erase the LLRs of data symbols in flagged cells (two bits per QPSK symbol)
        flagged = cells[:, self.data_sc // 12, self.data_sym]  # [B, num_data]
        flagged = flagged.repeat_interleave(self.tx._num_bits_per_symbol, dim=1)
        llr = llr.masked_fill(flagged[:, None, None, :], 0.0)

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
