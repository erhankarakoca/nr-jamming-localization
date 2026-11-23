"""Map-aware receiver: use the jamming map to protect decoding.

The standard receiver trusts every pilot and every data symbol. With a jamming map the gNB can
treat the flagged cells differently. The channel is assumed constant over the slot, so all DMRS
symbols contribute to one LMMSE estimate. Repeated observations of a subcarrier share its prior
correlation and have separate noise variances. Two ways to use the map are implemented:

**hard** (as in the classic literature, e.g. Li et al., MILCOM 2012):

1. *Repair the channel estimate.* Pilots in flagged DMRS cells are dropped, and the
   channel on every subcarrier is interpolated from the clean pilots with an LMMSE filter. The
   filter uses the same channel model as the DMRS residual feature (delays spread uniformly up to
   ``tau_max``), so one model both detects the attack and repairs it.
2. *Erase jammed data.* The soft bits (LLRs) of data symbols in flagged cells are set to 0, so
   the LDPC decoder treats them as unknown instead of trusting wrong values.

**soft**: the map says *where*, the receiver estimates *how strong* from the flagged cells:

1. Pilots of a flagged DMRS cell are kept, but the LMMSE filter is told their noise is larger: the
   cell's residual ``R`` (about 1 without jamming, ``1 + JNR`` under jamming) multiplies their
   noise variance. Strongly jammed pilots are effectively ignored; a false alarm (``R`` about 1)
   costs nothing.
2. The jamming power on the flagged data cells of each PRB is estimated from their excess
   received power (received energy minus the expected signal and noise energy), and added to the
   noise variance of those symbols before demapping. The LLRs shrink by exactly the right amount:
   to almost 0 under strong jamming, hardly at all after a false alarm.

**dmrs** (no map, what a receiver can do with its DMRS alone): a DMRS cell is declared interfered if its
DMRS residual exceeds the analytic 1% threshold (``jamloc.theory``); its pilots are down-weighted
as in *soft*, and the mean interference across its DMRS symbols is assumed to last the whole slot,
so all data symbols of the PRB get the extra noise variance. This is blind to data-only jamming, and it
wrongly penalizes the clean data of a pilot-jammed PRB.

Two more references without a map separate what the map adds from what the rest of the receiver
adds:

**lmmse**: the same LMMSE interpolation from all pilots, with the noise variance of the standard
receiver, and plain demapping. It differs from the standard receiver (Sionna's LS estimate with
linear interpolation) only in the interpolator.

**slot**: an interference-aware receiver without per-PRB localization: each pilot's variance is
multiplied by the mean residual over the allocation on its own DMRS symbol, and the mean excess
energy over all data symbols of the slot is added to the variance of every data symbol.

Why LMMSE and not a least-squares fit: dropping a block of pilots leaves a gap, and a plain fit
across the gap amplifies the noise enormously. LMMSE also gives an error variance for every
subcarrier. Where the gap is wider than the channel's coherence bandwidth, that variance
approaches the channel power, the equalizer reports a large effective noise, and the LLRs there
shrink towards zero by themselves: repair turns smoothly into erasure.

If the hard map flags every DMRS cell there is no clean pilot to interpolate from. The receiver
then uses Sionna's standard channel estimate, exactly like the unprotected receiver, and only
erases the flagged data cells.

**Data-aided passes** (``iterations > 0``, any mode): when the jammer covers every pilot, no map can
give the receiver a clean pilot, and even the true map leaves many blocks undecodable. But the data
cells the map calls clean are themselves observations of the channel, once their symbols are
known. After a decoding attempt, the LDPC decoder's soft output on every coded bit gives the mean
and variance of every transmitted QPSK symbol. Each data cell then becomes a *virtual pilot* whose
noise variance includes the remaining symbol uncertainty and, in the soft and hard modes, the
jamming the map reports for that cell (a flagged data cell carries the jammer and is left out).
Because the channel is constant over the slot, all observations of one subcarrier, the real pilots
with their own variances and the virtual ones, combine into one value per subcarrier
(inverse-variance weighting, a sufficient statistic), and the same LMMSE filter smooths them
across frequency. The slot is equalized and decoded again. A block keeps the first decoding whose
CRC passes, as a real receiver can, since the CRC tells it which attempt succeeded.
"""

import math

import numpy as np
import torch
from sionna.phy import Block
from sionna.phy.fec.ldpc import LDPC5GDecoder
from sionna.phy.mapping import Demapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.nr import LayerDemapper, PUSCHLSChannelEstimator, PUSCHTransmitter, TBDecoder
from sionna.phy.ofdm import LMMSEEqualizer

from jamloc import theory
from jamloc.config import SimConfig
from jamloc.features import dmrs_ls, dmrs_residuals, residual_per_prb
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
    Repeated indices in ``pilot_sc`` are observations of the same channel on different symbols.
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


MODES = ("hard", "soft", "dmrs", "lmmse", "slot", "cell")


def cell_threshold(num_re: int, alpha: float) -> float:
    """Threshold on the normalized energy of a cell of ``num_re`` resource elements whose
    remodulated residuals are independent circular Gaussians of known variance: their mean energy
    divided by its expectation is Gamma(num_re, 1 / num_re), exceeded with probability ``alpha``."""
    from scipy.stats import gamma

    return float(gamma.ppf(1 - alpha, a=num_re, scale=1 / num_re))


class SoftTBDecoder(Block):
    """Posterior LLRs of every coded bit of a transport block, in the order of the decoder's input.

    The same steps as Sionna's ``TBDecoder`` (descrambling, code-block deinterleaving, LDPC belief
    propagation), but the decoder returns soft values on the whole codeword, which are then put
    back into the transmitted order and scrambled again. Its input and output are LLRs
    ``log(P(b=1) / P(b=0))`` of shape ``[..., n]``.
    """

    def __init__(self, decoder: TBDecoder, **kwargs):
        super().__init__(**kwargs)
        self.tb = decoder
        encoder = decoder._tb_encoder
        self.n = encoder.n
        self.num_cbs = encoder.num_cbs
        self.cb_n = encoder.ldpc_encoder.n
        self.ldpc = LDPC5GDecoder(
            encoder=encoder.ldpc_encoder, num_iter=20, hard_out=False, return_infobits=False
        )
        self.perm_inv = decoder._output_perm_inv
        self.perm = torch.argsort(self.perm_inv)  # undoes the deinterleaving

    def call(self, llr: torch.Tensor) -> torch.Tensor:
        shape = llr.shape
        num_tx = self.tb._tb_encoder.num_tx
        x = llr.float().reshape(-1, num_tx, self.n)  # the layout Sionna's descrambler expects
        if self.tb._descrambler is not None:
            x = self.tb._descrambler(x)
        fillers = x.new_zeros(*x.shape[:-1], self.num_cbs * self.cb_n - self.n)
        x = torch.cat([x, fillers], dim=-1).index_select(-1, self.perm_inv)
        post = self.ldpc(x.reshape(-1, num_tx, self.num_cbs, self.cb_n))
        post = post.reshape(-1, num_tx, self.num_cbs * self.cb_n)
        post = post.index_select(-1, self.perm.to(post.device))[..., : self.n]
        if self.tb._descrambler is not None:  # scrambling flips signs: applying it again undoes it
            post = self.tb._descrambler(post)
        return post.reshape(shape)


def soft_qpsk(llr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of QPSK symbols from the LLRs of their two bits ``[..., num_sym, 2]``.

    Sionna's QPSK maps bit 0 to the real part and bit 1 to the imaginary part, a 1 to the negative
    sign, and its LLRs are ``log(P(1) / P(0))``, so the mean of each part is
    ``-tanh(LLR / 2) / sqrt(2)``. The symbols have unit energy, so their variance is
    ``1 - |mean|^2``.
    """
    m = -torch.tanh(llr.clamp(-30, 30) / 2) / math.sqrt(2)
    mean = torch.complex(m[..., 0], m[..., 1])
    return mean, (1 - mean.abs() ** 2).clamp_min(0)


class MapAwareReceiver(Block):
    """Receiver for one allocation size that uses a jamming map ``cells`` ``[B, n, 14]`` (bool).

    Same chain and outputs as ``PUSCHReceiverChain``; only the channel estimate and the LLRs differ.
    """

    def __init__(self, tx: PUSCHTransmitter, cfg: SimConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tx = tx
        self.cfg = cfg or SimConfig()
        rg = tx.resource_grid
        self.dmrs_symbols = tuple(rg.pilot_pattern.mask[0, 0].any(-1).nonzero().flatten().tolist())
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
        self.soft_decoder = SoftTBDecoder(self.decoder)
        self.data_sym, self.data_sc = data_positions(tx)
        self._cellwise = False  # set per call (see ``call``)

    def call(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
        cells: torch.Tensor | None,
        decode: bool = True,
        mode: str = "hard",
        iterations: int = 0,
        history: bool = False,
        cellwise: bool = False,
        refine: bool = False,
        renoise: bool = False,
        alpha: float = 0.01,
    ):
        """``mode``: ``"hard"``, ``"soft"`` (both use ``cells``), or ``"dmrs"``, ``"lmmse"``,
        ``"slot"``, ``"cell"`` (no map, ``cells`` is ignored). ``iterations``: data-aided passes
        after the first decoding (they need ``decode``). With ``history=True``, return a list of
        pass records starting with that first attempt. Each contains ``h_sc`` and
        ``err_var`` on the first DMRS symbol, ``x_hat``, ``llr``, and the
        individual pass's ``tb_crc`` (None with ``decode=False``).

        Options of the soft mode and its passes:

        * ``cellwise``: estimate the jamming power of a flagged data cell from its own resource
          elements instead of averaging it over the flagged data cells of its PRB.
        * ``refine``: after each failed decoding, test every data cell on its remodulated residual
          ``y - h_hat x_bar`` (``_refine``) and add the cells that fail it to the map, which then
          also has per-cell jamming estimates. The detector's map is the starting point; decoding
          adds what it missed.
        * ``renoise``: after each failed decoding, estimate the noise power again from the
          remodulated residual of the confidently decoded data cells the map calls clean, and
          recompute everything that depends on it (pilot variances, residuals, LLRs).
        """
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
        if mode == "cell":  # no map: every data cell gets its own jamming estimate
            cellwise = True
        if refine and mode != "soft":
            raise ValueError("refine needs the soft mode")
        if renoise and mode not in ("soft", "cell"):
            raise ValueError("renoise needs the soft or the cell mode")
        self._cellwise = cellwise

        # 1. Stack the LS observations in symbol order. The frequency indices repeat, so
        # indexing the prior by them gives the same frequency correlation within and between
        # DMRS symbols. Each observation still has its own diagonal noise variance.
        h_ls, pilot_sc, pilot_power = dmrs_ls(y, self.tx)
        clean, pilot_var, r, interfered = self._pilots(h_ls, pilot_sc, pilot_power, no, cells, mode)

        # 2. Repair: LMMSE interpolation, the same on all 14 symbols
        h_sc, err_sc = lmmse_interpolation(
            h_ls, pilot_sc, clean, pilot_var, num_sc, cfg.scs, cfg.tau_max
        )
        h_hat, err_var = self._on_grid(h_sc, err_sc, y)

        # 3. No clean pilot at all: nothing to interpolate from, use the standard estimate
        no_clean = ~clean.any(dim=1)
        if no_clean.any():
            h_std, err_std = self.standard_estimator(y, no)
            pick = no_clean[:, None, None, None, None, None, None]
            h_hat = torch.where(pick, h_std, h_hat)
            err_var = torch.where(pick, err_std.expand(h_hat.shape), err_var)

        # 4.-5. Equalize, protect the data symbols, demap
        x_hat, no_eff, llr, extra = self._detect(
            y, no, h_hat, err_var, h_sc, err_sc, cells, r, interfered, mode
        )

        b_hat = tb_crc = None
        if decode:
            b_hat, tb_crc = self.decoder(self.layer_demapper(llr))
        if history:
            # Read the estimate from the grid so a hard-mode fallback is captured as well.
            dmrs = self.dmrs_symbols[0]
            passes = [
                {
                    "h_sc": h_hat[:, 0, 0, 0, 0, dmrs],
                    "err_var": err_var[:, 0, 0, 0, 0, dmrs],
                    "x_hat": x_hat[:, 0, 0],
                    "llr": llr,
                    "tb_crc": tb_crc.flatten() if decode else None,
                }
            ]
        # 6. Data-aided passes: the decoded symbols of the clean data cells become pilots
        for _ in range(iterations if decode else 0):
            if tb_crc.all():
                break
            post = self.soft_decoder(self.layer_demapper(llr))
            mean, var = soft_qpsk(post.reshape(y.shape[0], -1, self.tx._num_bits_per_symbol))
            if refine:
                cells = self._refine(y, no, h_hat, err_var, mean, var, cells, alpha)
            if refine or renoise:  # the jamming the map now implies, with the old channel
                extra = self._detect(
                    y, no, h_hat, err_var, h_sc, err_sc, cells, r, interfered, mode
                )[3]
            h_new, err_new = self._data_aided(
                y, no, h_ls, pilot_sc, clean, pilot_var, h_hat, mean, var, extra
            )
            if renoise:
                # The data-aided estimate rests on many symbols per subcarrier, so its error is
                # small; the noise power is estimated around it, and everything that depends on
                # the noise power is computed again
                no = self._renoise(y, no, *self._on_grid(h_new, err_new, y), mean, var, cells, mode)
                clean, pilot_var, r, interfered = self._pilots(
                    h_ls, pilot_sc, pilot_power, no, cells, mode
                )
                extra = self._detect(
                    y, no, h_hat, err_var, h_sc, err_sc, cells, r, interfered, mode
                )[3]
                h_new, err_new = self._data_aided(
                    y, no, h_ls, pilot_sc, clean, pilot_var, h_hat, mean, var, extra
                )
            h_sc, err_sc = h_new, err_new
            h_hat, err_var = self._on_grid(h_sc, err_sc, y)
            x_hat, no_eff, llr, extra = self._detect(
                y, no, h_hat, err_var, h_sc, err_sc, cells, r, interfered, mode
            )
            b_new, crc_new = self.decoder(self.layer_demapper(llr))
            if history:
                # How sure the decoder was of each data cell's symbol (|mean|^2), as used by this
                # pass: zero for the cells it left out
                certainty = mean.abs() ** 2
                certainty = torch.where((certainty > 0.01) & torch.isfinite(extra), certainty, 0.0)
                passes.append(
                    {
                        "h_sc": h_sc,
                        "err_var": err_sc,
                        "x_hat": x_hat[:, 0, 0],
                        "llr": llr,
                        "tb_crc": crc_new.flatten(),
                        "certainty": certainty,
                    }
                )
            b_hat = torch.where(tb_crc[..., None], b_hat, b_new)  # keep a block once it passed
            tb_crc = tb_crc | crc_new

        out = {
            "h_hat": h_hat,
            "err_var": err_var,
            "x_hat": x_hat[:, 0, 0],
            "no_eff": no_eff[:, 0, 0],
            "llr": llr,
        }
        if decode:
            out["b_hat"], out["tb_crc"] = b_hat, tb_crc
        if history:
            out["history"] = passes
        return out

    def decoded_symbols(self, b_hat: torch.Tensor) -> torch.Tensor:
        """Data symbols ``[B, num_data]`` that the decoded information bits ``b_hat`` map to, in
        the order of ``data_positions`` (encode, scramble, map again, as the UE did)."""
        with torch.no_grad():
            coded = self.tx._tb_encoder(b_hat.reshape(-1, 1, b_hat.shape[-1]))
            return self.tx._mapper(coded)[:, 0]

    def known_symbol_residuals(self, y, no, x_data) -> torch.Tensor:
        """Residual ``[B, n, num_symbols]`` of every scheduled PRB on every data OFDM symbol, with
        the data symbols ``x_data`` ``[B, num_data]`` used as known symbols (NaN on symbols
        without data).

        Once a block passes its CRC, its data symbols are known exactly. On a data symbol, the
        LS values ``y / x`` of all subcarriers of the allocation then play the role of the DMRS
        pilots, and the same delay-subspace projection gives a residual per PRB with twelve known
        symbols instead of six (``features.residual_per_prb``). Without jamming its law is that
        of the DMRS residual with the projector of all subcarriers (``theory.weights`` with
        ``step=1``)."""
        num_prb = y.shape[-1] // 12
        out = y.real.new_full((y.shape[0], num_prb, self.cfg.num_symbols), float("nan"))
        for sym in sorted(set(self.data_sym.tolist())):
            on = self.data_sym == sym
            sc = self.data_sc[on]
            h_ls = y[:, 0, 0, sym, sc] / x_data[:, on.to(x_data.device)]
            power = float((x_data[:, on.to(x_data.device)].abs() ** 2).mean())
            out[:, :, sym] = residual_per_prb(h_ls, sc, no, power, self.cfg)
        return out

    def slot_residuals(self, y, no, x_data, exclude=None, order=1):
        """Residual ``[B, n, num_symbols]`` of every data and DMRS cell of a decoded slot, and
        the number of known symbols in each cell (NaN and 0 where a cell has none).

        With the data symbols ``x_data`` ``[B, num_data]`` known, every data and pilot resource
        element gives an LS value ``y / x = h + w / x``. The channel of a subcarrier changes
        little within a slot, so it is predicted for resource element ``(k, l)`` by a weighted
        least-squares fit of a constant (``order=0``) or a straight line in time (``order=1``)
        to the LS values of subcarrier ``k`` on the other data symbols, leaving out the cells in
        ``exclude`` ``[B, n, num_symbols]`` (cells known to be jammed, so that their energy does
        not leak into the prediction). The pilots are tested against the data but never used to
        predict. The prediction of ``(k, l)`` does not contain ``y_{k,l}``, so without jamming the
        difference is circular Gaussian with variance ``N0 / |x|^2`` plus the variance of the
        fit. The residual of a cell is the energy of its differences divided by the sum of these
        variances, Gamma(m, 1/m) for m known symbols of equal energy (``cell_threshold``)."""
        num_prb, width = y.shape[-1] // 12, y.shape[-1]
        num_sym = self.cfg.num_symbols
        dev, rows = y.device, y.shape[0]
        h = y.new_zeros(rows, num_sym, width)
        energy = y.real.new_zeros(rows, num_sym, width)  # |x|^2 of the known symbols
        sym, sc = self.data_sym.to(dev), self.data_sc.to(dev)
        x = x_data.to(dev)
        h[:, sym, sc] = y[:, 0, 0, sym, sc] / x
        energy[:, sym, sc] = x.abs() ** 2
        pp = self.tx.resource_grid.pilot_pattern
        psym, psc = torch.nonzero(pp.mask[0, 0], as_tuple=True)
        pilots = pp.pilots[0, 0]
        used = pilots.abs() > 0
        psym, psc, pilots = psym[used].to(dev), psc[used].to(dev), pilots[used].to(dev)
        h[:, psym, psc] = y[:, 0, 0, psym, psc] / pilots
        energy[:, psym, psc] = (pilots.abs() ** 2).to(energy.dtype)
        known = energy > 0
        predict = torch.zeros_like(known)
        predict[:, sym, sc] = True
        if exclude is not None:
            predict &= ~exclude.to(dev).transpose(1, 2).repeat_interleave(12, dim=2)
        w = energy * predict
        t = torch.arange(num_sym, device=dev, dtype=energy.dtype)[None, :, None]

        def others(v):  # sum over the other symbols of the same subcarrier
            return v.sum(1, keepdim=True) - v

        s0, sy = others(w), others(w * h)
        if order == 0:
            ok = s0 > 0.5
            pred = sy / s0.clamp_min(1e-12)
            pvar = 1 / s0.clamp_min(1e-12)
        else:
            s1, s2, sty = others(w * t), others(w * t * t), others(w * t * h)
            det = s0 * s2 - s1**2
            ok = det > 1e-6
            det = det.clamp_min(1e-12)
            pred = ((s2 * sy - s1 * sty) + t * (s0 * sty - s1 * sy)) / det
            pvar = (s2 - 2 * t * s1 + t * t * s0) / det
        var = torch.where(known, 1 / energy.clamp_min(1e-12) + pvar, 0.0)
        err = torch.where(known, (h - pred).abs() ** 2, 0.0)
        cells = (rows, num_sym, num_prb, 12)
        count = known.reshape(cells).sum(-1)
        fine = (ok | ~known).reshape(cells).all(-1) & (count > 0)
        r = err.reshape(cells).sum(-1) / (no[:, None, None] * var.reshape(cells).sum(-1))
        r = torch.where(fine, r, torch.nan)
        return r.transpose(1, 2), count.transpose(1, 2)

    def decoded_cell_test(self, y, no, x_data, alpha=0.01, peel=1e-6, exclude=None, order=1):
        """Jammed cells ``[B, n, num_symbols]`` of a decoded slot (data and DMRS cells), their
        residuals and their numbers of known symbols (``slot_residuals``).

        A strongly jammed data cell would leak into the prediction of every other cell of its
        subcarriers. So outliers are peeled off one at a time. In each round, the data cell with
        the largest residual in each PRB row is left out of the predictions if its residual
        exceeds the ``peel`` quantile of its null law, until no such cell is left. All cells are
        then tested at the ``alpha`` quantile. A cell with nothing left to predict it keeps the
        residual of the last round in which it had a prediction. ``exclude``: data cells left out
        of the predictions from the start (e.g. a detector's map)."""
        num_prb = y.shape[-1] // 12
        data = torch.zeros(num_prb, self.cfg.num_symbols, dtype=torch.bool, device=y.device)
        data[:, sorted(set(self.data_sym.tolist()))] = True
        excl = data.new_zeros((y.shape[0], num_prb, self.cfg.num_symbols))
        if exclude is not None:
            excl = excl | (exclude.to(excl.device) & data)
        last, count = None, None
        for _ in range(self.cfg.num_symbols + 1):
            r, count = self.slot_residuals(y, no, x_data, excl, order)
            last = r if last is None else torch.where(torch.isnan(r), last, r)
            high = self._null_quantile(count, peel)
            score = torch.nan_to_num(r / high, nan=-1.0).masked_fill(excl | ~data, -1.0)
            top = score.max(dim=2, keepdim=True).values
            new = (score == top) & (top > 1)
            if not new.any():
                break
            excl = excl | new
        return torch.nan_to_num(last, nan=0.0) > self._null_quantile(count, alpha), last, count

    @staticmethod
    def _null_quantile(count: torch.Tensor, alpha: float) -> torch.Tensor:
        """Threshold of Gamma(m, 1/m) exceeded with probability ``alpha``, for each cell's number
        of known symbols ``m`` (infinite where a cell has none)."""
        out = torch.full(count.shape, math.inf, device=count.device)
        for m in count.unique().tolist():
            if m > 0:
                out[count == m] = cell_threshold(int(m), alpha)
        return out

    def _pilots(self, h_ls, pilot_sc, pilot_power, no, cells, mode):
        """Which pilots the filter uses (``clean``), their LS noise variances, and the residuals
        and interference decisions of the DMRS cells (None where the mode has none)."""
        cfg = self.cfg
        num_dmrs = len(self.dmrs_symbols)
        num_prb = int(self.data_sc.max()) // 12 + 1
        pilot_prb = pilot_sc // 12
        pilot_dmrs = torch.arange(num_dmrs, device=h_ls.device).repeat_interleave(
            pilot_sc.numel() // num_dmrs
        )
        s2 = no / pilot_power  # LS noise variance per pilot without jamming
        clean = torch.ones_like(h_ls, dtype=torch.bool)
        pilot_var, r, interfered = s2, None, None
        if mode in ("hard", "soft"):
            flagged = cells[:, :, self.dmrs_symbols].transpose(1, 2)  # [B, num_dmrs, n]
        if mode == "hard":
            clean = ~flagged[:, pilot_dmrs, pilot_prb]
        elif mode != "lmmse":
            r = dmrs_residuals(h_ls, pilot_sc, no, pilot_power, cfg, num_dmrs).clamp_min(1)
            if mode == "slot":  # one allocation-wide level per DMRS symbol
                r = r.mean(dim=-1, keepdim=True).expand_as(r)
                interfered = torch.ones_like(r, dtype=torch.bool)
            elif mode in ("dmrs", "cell"):  # interfered: residual above the analytic 1% threshold
                thr = torch.tensor(
                    [theory.threshold(num_prb, b, 0.01, cfg) for b in range(num_prb)],
                    device=r.device,
                    dtype=r.dtype,
                )
                interfered = r > thr
            else:
                interfered = flagged
            inflate = torch.where(interfered, r, torch.ones_like(r))
            pilot_var = s2[:, None] * inflate[:, pilot_dmrs, pilot_prb]
        return clean, pilot_var, r, interfered

    def _cell_sum(self, values: torch.Tensor) -> torch.Tensor:
        """Sum of ``values`` ``[B, num_data]`` over the data resource elements of each cell
        (PRB and OFDM symbol), returned on every resource element of that cell."""
        cell = ((self.data_sc // 12) * self.cfg.num_symbols + self.data_sym).to(values.device)
        total = values.new_zeros(values.shape[0], int(cell.max()) + 1).index_add_(1, cell, values)
        return total[:, cell]

    def _cell_mean(self, values: torch.Tensor) -> torch.Tensor:
        return self._cell_sum(values) / self._cell_sum(torch.ones_like(values[:1]))

    def _remodulated(self, y, no, h_hat, err_var, mean, var):
        """Energy of ``y - h_hat x_bar`` on every data resource element, and its expected value
        without jamming. With ``x = x_bar + d``, ``E|d|^2 = var`` and ``|x_bar|^2 + var = 1``,
        ``y - h_hat x_bar = (h - h_hat) x_bar + h d + w`` has the expected energy
        ``err_var + |h_hat|^2 var + no``."""
        y_d = y[:, 0, 0, self.data_sym, self.data_sc]
        h_d = h_hat[:, 0, 0, 0, 0, self.data_sym, self.data_sc]
        e_d = err_var[:, 0, 0, 0, 0, self.data_sym, self.data_sc]
        energy = (y_d - h_d * mean).abs() ** 2
        return energy, e_d + h_d.abs() ** 2 * var, no[:, None]

    def _refine(self, y, no, h_hat, err_var, mean, var, cells, alpha):
        """The map after one decoding attempt: the detector's cells plus every data cell whose
        remodulated residual energy, normalized by its expected value without jamming, exceeds
        the ``alpha`` quantile of Gamma(m, 1/m) (``cell_threshold``, m resource elements)."""
        energy, model, noise = self._remodulated(y, no, h_hat, err_var, mean, var)
        stat = self._cell_sum(energy) / self._cell_sum(model + noise)
        num_re = int(self._cell_sum(torch.ones_like(energy[:1])).max())
        new = stat > cell_threshold(num_re, alpha)
        cells = cells.clone()
        prb = (self.data_sc // 12).to(cells.device)
        sym = self.data_sym.to(cells.device)
        cells[:, prb, sym] = cells[:, prb, sym] | new
        return cells

    def _renoise(self, y, no, h_hat, err_var, mean, var, cells, mode):
        """Noise power ``[B]`` from the remodulated residual of the data symbols that the decoder
        is sure of (``|x_bar|^2 > 0.5``) in cells the map calls clean, around a data-aided channel
        estimate ``h_hat``: the value of ``N0`` for which the median of
        ``|y - h_hat x_bar|^2 / (|h_hat|^2 var + N0)`` equals ln 2, the median of a unit
        exponential. Bisection on a log scale; slots without such symbols keep ``no``."""
        energy, _, _ = self._remodulated(y, no, h_hat, err_var, mean, var)
        # The filter's error variance assumed the old noise power, so it is left out: the
        # data-aided estimate passed in is accurate enough for its error to be small
        h_d = h_hat[:, 0, 0, 0, 0, self.data_sym, self.data_sc]
        model = h_d.abs() ** 2 * var
        use = mean.abs() ** 2 > 0.5
        if mode == "soft":
            prb = (self.data_sc // 12).to(cells.device)
            use = use & ~cells[:, prb, self.data_sym.to(cells.device)]
        lo, hi = (no * 1e-3).log(), (no * 1e2).log()
        for _ in range(40):
            mid = (lo + hi) / 2
            ratio = energy / (model + mid.exp()[:, None])
            med = torch.nanmedian(torch.where(use, ratio, torch.nan), dim=1).values
            above = med > math.log(2)  # N0 too small
            lo, hi = torch.where(above, mid, lo), torch.where(above, hi, mid)
        found = use.sum(dim=1) >= 24
        return torch.where(found, ((lo + hi) / 2).exp(), no)

    def _on_grid(self, h_sc, err_sc, y):
        """One estimate per subcarrier, the same on all symbols, in Sionna's ``h_hat`` layout."""
        shape = (y.shape[0], 1, 1, 1, 1, self.cfg.num_symbols, y.shape[-1])
        h_hat = h_sc[:, None, None, None, None, None, :].expand(shape)
        return h_hat, err_sc[:, None, None, None, None, None, :].expand(shape)

    def _detect(self, y, no, h_hat, err_var, h_sc, err_sc, cells, r, interfered, mode):
        """Equalized symbols, their effective noise, the LLRs, and the extra noise variance of
        every data symbol that the map implies (infinite for an erased one)."""
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)
        data_prb = self.data_sc // 12
        bits = self.tx._num_bits_per_symbol
        if mode == "hard":
            llr = self.demapper(x_hat, no_eff)
            flagged = cells[:, data_prb, self.data_sym]
            erased = flagged.repeat_interleave(bits, dim=1)
            llr = llr.masked_fill(erased[:, None, None, :], 0.0)
            extra = torch.where(flagged, math.inf, 0.0)
        elif mode == "lmmse":
            llr = self.demapper(x_hat, no_eff)
            extra = torch.zeros_like(x_hat[:, 0, 0].real)
        else:
            extra = self._jamming_variance(y, no, h_sc, err_sc, cells, r, interfered, mode)
            # For one antenna the unbiased LMMSE output is y / h_hat and its effective noise is
            # (err_var + no) / |h_hat|^2; extra noise of variance v adds v / |h_hat|^2
            h2 = (h_hat[:, 0, 0, 0, 0, self.data_sym, self.data_sc].abs() ** 2).clamp_min(1e-12)
            no_eff = no_eff + (extra / h2)[:, None, None, :]
            llr = self.demapper(x_hat, no_eff)
        return x_hat, no_eff, llr, extra

    def _data_aided(self, y, no, h_ls, pilot_sc, clean, pilot_var, h_hat, mean, var, extra):
        """LMMSE channel estimate from the pilots and the data cells, whose symbols are known only
        through their decoded ``mean`` and ``var`` ``[B, num_data]``.

        A data cell with received value ``y = h x + w`` and a symbol of mean ``m`` gives
        ``y / m``, an unbiased observation of ``h`` with variance
        ``(|h|^2 var + no + extra) / |m|^2``: the symbol's uncertainty, the noise and the jamming
        the map assigns to the cell (``extra``, infinite for an erased cell). Cells with
        ``|m|^2 < 0.01`` (nearly unknown symbols) are left out. All observations of a subcarrier
        are combined with inverse-variance weights before the LMMSE filter smooths across
        frequency.
        """
        num_sc = y.shape[-1]
        y_d = y[:, 0, 0, self.data_sym, self.data_sc]
        h_d = h_hat[:, 0, 0, 0, 0, self.data_sym, self.data_sc]
        power = mean.abs() ** 2
        use = (power > 0.01) & torch.isfinite(extra)
        var_d = (
            h_d.abs() ** 2 * var + no[:, None] + torch.where(use, extra, 0.0)
        ) / power.clamp_min(0.01)
        w_d = torch.where(use, 1 / var_d, 0.0)
        obs_d = torch.where(use, y_d / torch.where(use, mean, torch.ones_like(mean)), 0)

        pilot_var = pilot_var if pilot_var.dim() == 2 else pilot_var[:, None].expand_as(h_ls.real)
        ok = clean & torch.isfinite(h_ls)
        w_p = torch.where(ok, 1 / pilot_var, 0.0)
        obs_p = torch.where(ok, h_ls, 0)

        rows = y.shape[0]
        num = y_d.new_zeros(rows, num_sc).index_add_(1, pilot_sc, (w_p * obs_p).to(y_d.dtype))
        num = num.index_add_(1, self.data_sc, (w_d * obs_d).to(y_d.dtype))
        den = w_d.new_zeros(rows, num_sc).index_add_(1, pilot_sc, w_p.to(w_d.dtype))
        den = den.index_add_(1, self.data_sc, w_d)
        h_obs = num / den.clamp_min(1e-12)
        every = torch.arange(num_sc, device=y.device)
        return lmmse_interpolation(
            h_obs,
            every,
            torch.ones_like(den, dtype=torch.bool),
            1 / den.clamp_min(1e-12),
            num_sc,
            self.cfg.scs,
            self.cfg.tau_max,
        )

    def _jamming_variance(self, y, no, h_sc, err_sc, cells, r, interfered, mode):
        """Estimated jamming power ``[B, num_data]`` on every data symbol (0 where none)."""
        data_prb = self.data_sc // 12
        if mode == "slot":  # mean excess energy over all data symbols, on every data symbol
            excess = self._excess_energy(y, no, h_sc, err_sc)
            return excess.mean(dim=1, keepdim=True).clamp_min(0).expand_as(excess)
        if mode == "dmrs":
            # Residual R = 1 + JNR. Average the estimated interference, including zero for
            # clean DMRS cells, before extending it to the PRB's data symbols.
            per_dmrs = torch.where(interfered, no[:, None, None] * (r - 1), torch.zeros_like(r))
            per_prb = per_dmrs.mean(dim=1)
            return per_prb[:, data_prb]

        excess = self._excess_energy(y, no, h_sc, err_sc)
        if mode == "cell":  # no map: every data cell's own excess energy
            return self._cell_mean(excess).clamp_min(0)
        flagged = cells[:, data_prb, self.data_sym]  # [B, num_data]
        if self._cellwise:  # each flagged cell from its own resource elements
            return torch.where(flagged, self._cell_mean(excess).clamp_min(0), 0.0)
        # Average over the flagged data REs of each PRB.
        num_prb = y.shape[-1] // 12
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
