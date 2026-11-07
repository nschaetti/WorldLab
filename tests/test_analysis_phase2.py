import numpy as np

from worldlab.board.analysis import (
    build_exponential_kernel_matrix,
    fit_exponential_profile,
    lattice_1d_concept,
    pairwise_distances,
    pca_from_gram,
    translation_symmetry_diagnostic,
)


def test_fit_exponential_profile_recovers_parameters_on_noiseless_data():
    rng = np.random.default_rng(0)
    d = np.sort(rng.uniform(0.0, 2.0, size=200))
    amp_true = 1.7
    sigma_true = 0.42
    offset_true = -0.2
    y = offset_true + amp_true * np.exp(-d / sigma_true)

    fit = fit_exponential_profile(distances=d, values=y, fit_offset=True)

    assert abs(fit.sigma - sigma_true) < 0.06
    assert abs(fit.amplitude - amp_true) < 0.15
    assert abs(fit.offset - offset_true) < 0.1
    assert fit.r2 > 0.999


def test_translation_symmetry_diagnostic_detects_distance_only_structure():
    concept = lattice_1d_concept(num_tokens=32, periodic=True, period=2.0)
    dist = pairwise_distances(concept)
    matrix = build_exponential_kernel_matrix(distances=dist, amplitude=1.0, sigma=0.35, offset=0.1)

    diag_good = translation_symmetry_diagnostic(matrix=matrix, distances=dist, num_bins=20)
    assert diag_good.r2 > 0.99

    rng = np.random.default_rng(1)
    shuffled = matrix.copy()
    perm = rng.permutation(shuffled.size)
    shuffled = shuffled.reshape(-1)[perm].reshape(shuffled.shape)
    diag_bad = translation_symmetry_diagnostic(matrix=shuffled, distances=dist, num_bins=20)
    assert diag_bad.r2 < 0.7


def test_pca_from_gram_reconstructs_centered_gram_for_positive_spectrum():
    concept = lattice_1d_concept(num_tokens=20, periodic=False)
    dist = pairwise_distances(concept)
    gram = build_exponential_kernel_matrix(distances=dist, amplitude=1.3, sigma=0.5, offset=0.2)

    coords, evals, _ = pca_from_gram(gram=gram, rank=None, center=True)
    recon = coords @ coords.T

    n = gram.shape[0]
    p = np.eye(n) - np.ones((n, n)) / n
    target = p @ gram @ p

    rel_err = np.linalg.norm(recon - target) / (np.linalg.norm(target) + 1e-12)
    assert evals.shape[0] > 0
    assert rel_err < 1e-8
