import numpy as np

from worldlab.board.analysis import (
    ablate_target_block,
    build_exponential_kernel_matrix,
    helper_reconstruction_curve,
    lattice_1d_concept,
    pairwise_distances,
    procrustes_error,
    reconstruct_target_with_helpers,
    select_helpers_by_coupling,
)


def _make_kernel_matrix(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    concept = lattice_1d_concept(num_tokens=n, periodic=True, period=2.0)
    dist = pairwise_distances(concept)
    matrix = build_exponential_kernel_matrix(distances=dist, amplitude=1.0, sigma=0.2, offset=0.05)
    return matrix, dist


def test_ablate_target_block_sets_target_submatrix_to_zero():
    m, _ = _make_kernel_matrix(24)
    target = np.array([0, 1, 2, 3])
    a = ablate_target_block(m, target_indices=target, value=0.0)

    assert np.allclose(a[np.ix_(target, target)], 0.0)
    # Off-target entries should remain unchanged.
    assert np.isclose(a[10, 11], m[10, 11])


def test_procrustes_error_is_near_zero_under_rotation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    y = x @ q

    err = procrustes_error(x, y)
    assert err < 1e-10


def test_helper_reconstruction_curve_improves_with_more_helpers_on_synthetic_kernel():
    m, _ = _make_kernel_matrix(64)
    target = np.arange(10)
    a = ablate_target_block(m, target_indices=target, value=0.0)

    # Smoke test helper selection and single reconstruction API.
    helpers = select_helpers_by_coupling(a, target_indices=target, top_k=8)
    coords, gram, used = reconstruct_target_with_helpers(
        matrix=a,
        target_indices=target,
        helper_indices=helpers,
        rank=10,
        mode="absolute",
    )
    assert coords.shape[0] == target.shape[0]
    assert gram.shape == (target.shape[0], target.shape[0])
    assert used.shape[0] == target.shape[0] + helpers.shape[0]

    curve = helper_reconstruction_curve(
        original_matrix=m,
        ablated_matrix=a,
        target_indices=target,
        helper_sizes=[0, 4, 8, 16],
        strategy="top_coupling",
        num_trials=5,
        rank=10,
        mode="absolute",
        seed=1,
    )

    assert curve.procrustes_mean.shape == (4,)
    assert curve.gram_rel_mse_mean.shape == (4,)
    # Expect quality to improve when adding many helpers vs no helper.
    assert curve.procrustes_mean[-1] < curve.procrustes_mean[0]
