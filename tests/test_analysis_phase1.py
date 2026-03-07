import numpy as np

from boardGPT.analysis import (
    PolicyConfig,
    SyntheticWorld,
    build_cooccurrence_statistics,
    generate_synthetic_sequences,
    lattice_1d_concept,
    lattice_nd_concept,
    pairwise_distances,
)


def test_cooccurrence_statistics_shapes_and_normalization():
    sequences = [
        [0, 1, 2, 1, 0],
        [2, 2, 1, 0],
    ]
    stats = build_cooccurrence_statistics(
        sequences=sequences,
        vocab_size=3,
        window_size=2,
        symmetric=True,
    )

    assert stats.counts_ij.shape == (3, 3)
    assert stats.p_ij.shape == (3, 3)
    assert stats.p_i.shape == (3,)
    assert stats.pmi.shape == (3, 3)
    assert stats.m_star.shape == (3, 3)
    assert np.isclose(np.sum(stats.p_ij), 1.0)
    assert np.all(stats.p_i >= 0)
    assert np.isclose(np.sum(stats.p_i), 1.0)


def test_periodic_distance_wraps_correctly_1d():
    concept = lattice_1d_concept(num_tokens=5, periodic=True, period=2.0)
    dist = pairwise_distances(concept)

    # Token 0 is at x=-1 and token 4 at x=+1, which are neighbors on period 2.
    assert np.isclose(dist[0, 4], 0.0)


def test_synthetic_generator_hard_history_radius_zero_stays_on_start_token():
    concept = lattice_nd_concept(shape=(4, 4), periodic=(False, False))
    world = SyntheticWorld(concept=concept)
    cfg = PolicyConfig(
        length=12,
        num_sequences=8,
        seed=7,
        transition_kernel="none",
        transition_strength=0.0,
        history_strength=1.0,
        history_hard_radius=0.0,
    )
    ds = generate_synthetic_sequences(world=world, config=cfg)

    assert ds.sequences.shape == (8, 12)
    # With history radius 0 and no transition term, only already-visited tokens
    # are allowed. Since only one token is visited at t=0, sequence stays constant.
    for row in ds.sequences:
        assert np.all(row == row[0])
