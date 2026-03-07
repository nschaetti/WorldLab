import numpy as np

from boardGPT.analysis import (
    PolicyConfig,
    SyntheticGPTTrainConfig,
    SyntheticWorld,
    build_tiny_gpt_for_synthetic,
    extract_mean_token_representations_from_model,
    generate_synthetic_sequences,
    lattice_1d_concept,
    pairwise_distances,
    representation_symmetry_report,
    train_gpt_on_synthetic_sequences,
)


def _make_small_dataset():
    concept = lattice_1d_concept(num_tokens=12, periodic=True, period=2.0)
    world = SyntheticWorld(concept=concept)
    cfg = PolicyConfig(
        length=20,
        num_sequences=128,
        seed=0,
        transition_kernel="exponential",
        transition_scale=0.25,
        transition_strength=1.0,
        epsilon_uniform=0.05,
    )
    ds = generate_synthetic_sequences(world=world, config=cfg)
    return world, ds


def test_extract_mean_token_representations_shape_and_finiteness():
    world, ds = _make_small_dataset()
    model = build_tiny_gpt_for_synthetic(
        vocab_size=world.vocab_size,
        block_size=ds.sequences.shape[1] - 1,
        n_layer=2,
        n_head=2,
        n_embd=32,
        use_flash=False,
    )

    reps = extract_mean_token_representations_from_model(
        model=model,
        sequences=ds.sequences,
        layer=1,
        vocab_size=world.vocab_size,
        device="cpu",
        batch_size=64,
    )
    assert reps.shape == (world.vocab_size, model.config.n_embd)
    assert np.all(np.isfinite(reps))


def test_synthetic_training_loop_returns_finite_losses():
    world, ds = _make_small_dataset()
    model = build_tiny_gpt_for_synthetic(
        vocab_size=world.vocab_size,
        block_size=ds.sequences.shape[1] - 1,
        n_layer=2,
        n_head=2,
        n_embd=32,
        use_flash=False,
    )
    cfg = SyntheticGPTTrainConfig(
        batch_size=32,
        max_steps=20,
        learning_rate=5e-4,
        seed=0,
        device="cpu",
        log_interval=0,
    )
    losses = train_gpt_on_synthetic_sequences(model=model, sequences=ds.sequences, config=cfg)

    assert len(losses) == 20
    assert all(np.isfinite(x) for x in losses)
    # Weak sanity check: loss should not explode over short training.
    assert losses[-1] < losses[0] * 1.5


def test_representation_symmetry_report_runs_end_to_end():
    world, ds = _make_small_dataset()
    model = build_tiny_gpt_for_synthetic(
        vocab_size=world.vocab_size,
        block_size=ds.sequences.shape[1] - 1,
        n_layer=2,
        n_head=2,
        n_embd=32,
        use_flash=False,
    )

    reps = extract_mean_token_representations_from_model(
        model=model,
        sequences=ds.sequences,
        layer=1,
        vocab_size=world.vocab_size,
        device="cpu",
        batch_size=64,
    )
    dist = pairwise_distances(world.concept)
    report = representation_symmetry_report(
        representations=reps,
        distances=dist,
        pca_rank=4,
        num_bins=10,
    )

    assert report.bundle.gram.shape == (world.vocab_size, world.vocab_size)
    assert report.bundle.pca_coords.shape[0] == world.vocab_size
    assert np.isfinite(report.symmetry.r2)
