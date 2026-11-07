from worldlab.board.interpretability.attribution import (
    compute_intervention_attribution,
    compute_intervention_attribution_summary,
    compute_layer_cell_attribution_map,
    rank_top_contributors,
)


def test_compute_intervention_attribution_vector():
    rows = [
        {"target_cell": 2, "target_prob_gain": 0.4},
        {"target_cell": 2, "target_prob_gain": 0.2},
        {"target_cell": 8, "target_prob_gain": -0.1},
    ]
    vec = compute_intervention_attribution(rows)
    assert vec.shape == (64,)
    assert float(vec[2].item()) > 0.25


def test_layer_map_and_ranking():
    layer_results = {
        0: [
            {"target_cell": 0, "target_prob_gain": 0.1},
            {"target_cell": 1, "target_prob_gain": 0.2},
        ],
        1: [
            {"target_cell": 1, "target_prob_gain": 0.6},
            {"target_cell": 4, "target_prob_gain": 0.3},
        ],
    }
    amap = compute_layer_cell_attribution_map(layer_results)
    assert amap.shape == (2, 64)

    top = rank_top_contributors(amap, layers=[0, 1], top_k=3)
    assert len(top) == 3
    assert top[0]["layer"] in [0.0, 1.0]

    summary = compute_intervention_attribution_summary(layer_results, top_k=5)
    assert summary["attribution_map"].shape == (2, 64)
    assert len(summary["top_contributors"]) == 5
