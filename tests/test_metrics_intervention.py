import torch

from worldlab.board.interpretability.metrics_intervention import (
    aggregate_by_cell,
    compute_delta_norm_stats,
    compute_off_target_impact,
    compute_success_rate,
    compute_target_gain_stats,
)


def _fake_result(target_cell: int, gain: float, success: bool):
    before = torch.zeros(64, 3)
    after = before.clone()
    target_class = 2
    after[:, target_class] = 0.1
    after[target_cell, target_class] = 0.1 + gain
    return {
        "target_cell": target_cell,
        "target_class": target_class,
        "target_prob_gain": gain,
        "delta_norm": abs(gain) + 0.1,
        "success": success,
        "before_board_probs": before,
        "after_board_probs": after,
    }


def test_core_intervention_metrics():
    rows = [
        _fake_result(3, 0.4, True),
        _fake_result(3, 0.2, True),
        _fake_result(10, -0.1, False),
    ]
    assert compute_success_rate(rows) == 2 / 3

    gain = compute_target_gain_stats(rows)
    assert gain["mean"] > 0.1
    assert gain["max"] >= gain["min"]

    norm = compute_delta_norm_stats(rows)
    assert norm["mean"] > 0.0

    off = compute_off_target_impact(rows)
    assert off["num_valid_examples"] == 3.0


def test_aggregate_by_cell():
    rows = [
        _fake_result(0, 0.3, True),
        _fake_result(0, 0.2, True),
        _fake_result(1, -0.2, False),
    ]
    agg = aggregate_by_cell(rows)
    assert set(agg.keys()) == {0, 1}
    assert agg[0]["count"] == 2
    assert agg[0]["success_rate"] == 1.0
