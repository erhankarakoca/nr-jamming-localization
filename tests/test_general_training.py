import numpy as np

from jamloc.experiments.general_training import (
    CONDITIONS,
    TRAININGS,
    pilot_pd,
    pilot_pd_matched,
    table,
)
from jamloc.features import FEATURE_NAMES


def test_pilot_pd_picks_scheduled_pilot_cells_at_minus_15_db():
    rows = [
        {"attack": "pilot", "channel": "both", "cells": "all", "jsr_db": -15.0, "pd": 0.1},
        {"attack": "pilot", "channel": "both", "cells": "scheduled", "jsr_db": -12.0, "pd": 0.2},
        {"attack": "pilot", "channel": "both", "cells": "scheduled", "jsr_db": -15.0, "pd": 0.3},
        {"attack": "data", "channel": "both", "cells": "scheduled", "jsr_db": -15.0, "pd": 0.4},
    ]
    assert pilot_pd(rows) == 0.3
    assert np.isnan(pilot_pd(rows[3:]))  # no pilot attack, as on the pulsed grid


def test_matched_pilot_pd_uses_the_grids_own_clean_cells():
    # Two slots, each with 100 cells scheduled: slot 0 is clean, slot 1 a weak pilot attack whose
    # 10 jammed cells score 1 to 10. The clean cells score 0.00 to 0.99, so a 1% false-alarm rate
    # puts the threshold near 0.98 and every jammed cell above it.
    features = np.zeros((2, len(FEATURE_NAMES), 10, 10), dtype=np.float32)
    features[:, FEATURE_NAMES.index("alloc_mask")] = 1
    labels = np.zeros((2, 10, 10), dtype=np.uint8)
    labels[1, 0] = 1
    scores = np.zeros((2, 10, 10))
    scores[0] = np.arange(100).reshape(10, 10) / 100
    scores[1] = -np.inf
    scores[1, 0] = np.arange(1, 11)
    test = {
        "features": features,
        "labels": labels,
        "attack": np.array([0, 1]),
        "jsr_db": np.array([0.0, -15.0]),
    }
    assert pilot_pd_matched(scores, test) == 1.0
    test["jsr_db"] = np.array([0.0, -12.0])  # no weak pilot attack
    assert np.isnan(pilot_pd_matched(scores, test))


def test_table_has_one_row_per_grid_and_a_column_per_training():
    rows = []
    for condition in CONDITIONS.values():
        for training, _ in TRAININGS.values():
            pd = np.nan if "pulsed" in condition else 0.5
            rows.append(
                {
                    "condition": condition,
                    "training": training,
                    "f1_mean": 0.9,
                    "f1_std": 0.01,
                    "fa_mean": 0.012,
                    "pd_pilot_mean": pd,
                    "pd_pilot_matched_mean": pd,
                }
            )
    lines = table(rows).splitlines()
    assert len(lines) == 2 + len(CONDITIONS)
    assert lines[2].count("0.900 +- 0.010 / 1.2%") == len(TRAININGS)
    assert lines[2].endswith("| 0.50 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 |")
    assert lines[-1].endswith("| - | - |")  # pulsed: no pilot attack


def test_decoding_table_pools_the_seed_files(tmp_path):
    from jamloc.experiments.general_training import decoding_table

    run = tmp_path / "hop" / "recovery"
    run.mkdir(parents=True)
    head = "receiver,jammer,noise_estimate,seed,attack,jsr_db,snr_db,ber,bler,block_errors,blocks"
    for seed, errors in enumerate((3, 1)):
        rows = [
            f"standard,hopping,true,{seed},data,9.0,15.0,0,0,{errors},100",
            f"gen_shrink,hopping,true,{seed},data,9.0,15.0,0,0,0,100",
        ]
        (run / f"recovery_C_hopping_seed{seed}.csv").write_text("\n".join([head, *rows]) + "\n")
    text = decoding_table(tmp_path)
    assert "| Receiver | hopping data +9 |" in text
    assert "| No protection | 2.00 |" in text  # 4 block errors in 200 blocks
    assert "| General U-Net, shrinkage | 0/200 |" in text
    assert "| True map, shrinkage | - |" in text  # not decoded here
    assert "16-QAM" not in text
