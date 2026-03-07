import numpy as np

from boardGPT.analysis import (
    evaluate_linear_decoding_vs_rank,
    fit_linear_decoder,
    mse,
    predict_linear_decoder,
    relative_mse,
)


def test_fit_linear_decoder_exact_recovery_noiseless_square_system():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5))
    w_true = rng.normal(size=(5, 2))
    y = x @ w_true

    w = fit_linear_decoder(x=x, y=y, ridge_alpha=0.0)
    yhat = predict_linear_decoder(x, w)

    assert mse(y, yhat) < 1e-12
    assert relative_mse(y, yhat) < 1e-12


def test_evaluate_linear_decoding_vs_rank_shapes_and_basic_trend():
    rng = np.random.default_rng(1)
    n = 120
    r_max = 12
    x = rng.normal(size=(n, r_max))

    # Build target mostly from first 3 dimensions.
    w = np.zeros((r_max, 1), dtype=np.float64)
    w[0, 0] = 1.5
    w[1, 0] = -1.0
    w[2, 0] = 0.5
    y = x @ w + 0.01 * rng.normal(size=(n, 1))

    ranks = [1, 2, 3, 6, 10]
    res = evaluate_linear_decoding_vs_rank(
        features=x,
        targets=y,
        ranks=ranks,
        num_trials=20,
        train_size=60,
        ridge_alpha=0.0,
        seed=3,
    )

    assert res.ranks.shape == (len(ranks),)
    assert res.train_mse_mean.shape == (len(ranks),)
    assert res.test_mse_mean.shape == (len(ranks),)
    assert np.all(np.isfinite(res.test_rel_mse_mean))

    # Since target depends on first 3 dims, rank 3 should outperform rank 1.
    assert res.test_mse_mean[2] < res.test_mse_mean[0]


def test_ridge_regularization_path_runs_for_multid_targets():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(90, 8))
    w_true = rng.normal(size=(8, 3))
    y = x @ w_true + 0.1 * rng.normal(size=(90, 3))

    res = evaluate_linear_decoding_vs_rank(
        features=x,
        targets=y,
        ranks=[2, 4, 6, 8],
        num_trials=10,
        train_size=45,
        ridge_alpha=1e-2,
        seed=9,
    )

    assert res.ridge_alpha == 1e-2
    assert res.train_size == 45
    assert np.all(res.train_mse_mean >= 0)
    assert np.all(res.test_mse_mean >= 0)
