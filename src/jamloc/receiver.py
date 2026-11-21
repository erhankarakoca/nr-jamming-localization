"""PUSCH receiver written out step by step, as in the Sionna tutorials.

Sionna's ``PUSCHReceiver`` runs channel estimation, detection and decoding in one call. Here the
same chain is built from its blocks, and the LMMSE detector is split into an equalizer and a
demapper, so that the intermediate results the jamming features need (channel estimates,
equalized symbols, effective noise variances) are returned as well::

    y -> LS channel estimation -> LMMSE equalization -> demapping -> layer demapping -> decoding
"""

import numpy as np
import torch
from sionna.phy import Block
from sionna.phy.mapping import Demapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.nr import LayerDemapper, PUSCHLSChannelEstimator, PUSCHTransmitter, TBDecoder
from sionna.phy.ofdm import LMMSEEqualizer


def data_positions(tx: PUSCHTransmitter) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid position ``(ofdm_symbol, subcarrier)`` of each equalized data symbol.

    The equalizer returns the data symbols of the grid in row-major order (symbol by symbol),
    skipping the resource elements reserved for DMRS.
    """
    mask = tx.resource_grid.pilot_pattern.mask[0, 0]  # [num_symbols, num_subcarriers], 1 = DMRS
    sym, sc = torch.nonzero(mask == 0, as_tuple=True)
    return sym, sc


class PUSCHReceiverChain(Block):
    """Receiver for one allocation size. Input ``y`` ``[B, 1, 1, 14, 12 n]``, noise ``no`` ``[B]``.

    Returns a dict with ``h_hat``, ``err_var``, ``x_hat`` ``[B, num_data]``, ``no_eff``
    ``[B, num_data]``, ``llr`` and, if ``decode=True``, ``b_hat`` and ``tb_crc``.
    """

    def __init__(self, tx: PUSCHTransmitter, **kwargs):
        super().__init__(**kwargs)
        rg = tx.resource_grid
        sm = StreamManagement(np.ones([1, 1], bool), 1)  # one UE, one layer

        # Use the transmitter's DMRS configuration (including additional symbols), as
        # Sionna's own PUSCHReceiver does.
        self.channel_estimator = PUSCHLSChannelEstimator(
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

    def call(self, y: torch.Tensor, no: torch.Tensor, decode: bool = True) -> dict:
        # Channel estimation from the DMRS (the gNB only knows the thermal noise variance)
        h_hat, err_var = self.channel_estimator(y, no)

        # LMMSE equalization: equalized symbols and their effective noise variance
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)

        # Demapping to LLRs
        llr = self.demapper(x_hat, no_eff)

        out = {
            "h_hat": h_hat,
            "err_var": err_var,
            "x_hat": x_hat[:, 0, 0],  # single UE, single layer
            "no_eff": no_eff[:, 0, 0],
            "llr": llr,
        }
        if decode:
            # Layer demapping and transport block decoding (LDPC + CRC)
            b_hat, tb_crc = self.decoder(self.layer_demapper(llr))
            out["b_hat"], out["tb_crc"] = b_hat, tb_crc
        return out
