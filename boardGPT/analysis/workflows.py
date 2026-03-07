"""
High-level reproducible workflows for symmetry experiments (Phase 6).

This module provides a compact end-to-end runner combining Phases 1-5 into a
single callable function for quick iteration and reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
import json

import numpy as np

from .ablations import ablate_target_block, helper_reconstruction_curve
from .cooccurrence import build_cooccurrence_statistics
from .concepts import lattice_1d_concept, lattice_nd_concept, pairwise_distances
from .decoding import evaluate_linear_decoding_vs_rank
from .representations import extract_mean_token_representations_from_model, representation_symmetry_report
from .synthetic_gpt import (
    SyntheticGPTTrainConfig,
    build_tiny_gpt_for_synthetic,
    train_gpt_on_synthetic_sequences,
)
from .synthetic_sequences import PolicyConfig, SyntheticWorld, generate_synthetic_sequences
from .theory_karkada import fit_exponential_kernel_from_matrix, pca_from_gram, translation_symmetry_diagnostic


@dataclass(frozen=True)
class WorkflowConfig:
    """
    End-to-end workflow configuration.

    Attributes:
        world_shape: Token lattice shape (e.g., ``(32,)``, ``(8,8)``, ``(4,4,4)``).
        periodic: Optional periodic mask per axis. If None, open boundaries.
        world_name: Human-readable world name.
        policy: Synthetic data policy config.
        cooc_window: Co-occurrence context window.
        symmetry_bins: Number of bins in distance-profile diagnostics.
        run_model: If True, train tiny GPT and analyze representations.
        model_train: Tiny GPT train config.
        model_layer: Layer index used for representation extraction.
        ablation_target_size: Number of target tokens for helper reconstruction.
        ablation_helper_sizes: Helper-size curve points.
        decoding_max_rank: Maximum rank used in decoding curve.
        decoding_trials: Number of decoding trials.
        output_dir: Optional output directory to persist summary JSON.
    """

    world_shape: tuple[int, ...] = (32,)
    periodic: Optional[tuple[bool, ...]] = None
    world_name: str = "workflow_world"
    policy: PolicyConfig = PolicyConfig(length=48, num_sequences=1200, seed=0)
    cooc_window: int = 8
    symmetry_bins: int = 24
    run_model: bool = True
    model_train: SyntheticGPTTrainConfig = SyntheticGPTTrainConfig(max_steps=60, batch_size=64, log_interval=30)
    model_layer: int = 1
    ablation_target_size: int = 8
    ablation_helper_sizes: tuple[int, ...] = (0, 2, 4, 8, 12)
    decoding_max_rank: int = 12
    decoding_trials: int = 20
    output_dir: Optional[str] = None


def _json_compatible(obj: Any) -> Any:
    """Convert nested objects/arrays to JSON-compatible primitives."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_json_compatible(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_compatible(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _json_compatible(v) for k, v in obj.__dict__.items()}
    return str(obj)


def _make_world(shape: tuple[int, ...], periodic: Optional[tuple[bool, ...]], name: str) -> SyntheticWorld:
    if len(shape) == 1:
        concept = lattice_1d_concept(num_tokens=shape[0], periodic=bool(periodic[0]) if periodic else False, name=name)
    else:
        concept = lattice_nd_concept(shape=shape, periodic=periodic, name=name)
    return SyntheticWorld(concept=concept)


def run_synthetic_symmetry_workflow(cfg: WorkflowConfig) -> dict[str, Any]:
    """
    Run an end-to-end synthetic symmetry workflow (Phases 1-5).

    Returns:
        Dictionary summary with co-occurrence diagnostics, optional model
        diagnostics, decoding curves, and helper-ablation curves.
    """
    world = _make_world(shape=cfg.world_shape, periodic=cfg.periodic, name=cfg.world_name)
    dist = pairwise_distances(world.concept)

    dataset = generate_synthetic_sequences(world=world, config=cfg.policy)
    stats = build_cooccurrence_statistics(
        sequences=dataset.sequences,
        vocab_size=world.vocab_size,
        window_size=cfg.cooc_window,
    )

    cooc_sym = translation_symmetry_diagnostic(
        matrix=stats.m_star,
        distances=dist,
        num_bins=cfg.symmetry_bins,
    )
    cooc_fit = fit_exponential_kernel_from_matrix(
        matrix=stats.m_star,
        distances=dist,
        num_bins=cfg.symmetry_bins,
        fit_offset=True,
    )

    # Phase 5 helper ablation on co-occurrence matrix.
    target_size = min(cfg.ablation_target_size, world.vocab_size)
    target = np.arange(target_size, dtype=np.int64)
    target_list = target.tolist()
    ablated = ablate_target_block(stats.m_star, target_indices=target_list, value=0.0)
    helper_pool_size = max(world.vocab_size - target_size, 0)
    helper_sizes = tuple(sorted({int(h) for h in cfg.ablation_helper_sizes if int(h) >= 0 and int(h) <= helper_pool_size}))
    if len(helper_sizes) == 0:
        helper_sizes = (0,)

    helper_curve = helper_reconstruction_curve(
        original_matrix=stats.m_star,
        ablated_matrix=ablated,
        target_indices=target_list,
        helper_sizes=helper_sizes,
        strategy="top_coupling",
        num_trials=10,
        rank=min(target_size, 12),
        mode="absolute",
        seed=cfg.policy.seed,
    )

    summary: dict[str, Any] = {
        "world": {
            "shape": cfg.world_shape,
            "periodic": cfg.periodic,
            "vocab_size": world.vocab_size,
            "dim": world.concept.dim,
        },
        "cooccurrence": {
            "symmetry_r2": cooc_sym.r2,
            "symmetry_mse": cooc_sym.mse,
            "fit": asdict(cooc_fit),
        },
        "ablation": {
            "target_size": int(target_size),
            "helper_sizes": helper_curve.helper_sizes,
            "procrustes_mean": helper_curve.procrustes_mean,
            "gram_rel_mse_mean": helper_curve.gram_rel_mse_mean,
        },
    }

    if cfg.run_model:
        seq_len = dataset.sequences.shape[1]
        block_size = max(seq_len - 1, 1)
        model = build_tiny_gpt_for_synthetic(
            vocab_size=world.vocab_size,
            block_size=block_size,
            n_layer=max(cfg.model_layer + 1, 2),
            n_head=2,
            n_embd=64,
            use_flash=False,
        )
        losses = train_gpt_on_synthetic_sequences(model=model, sequences=dataset.sequences, config=cfg.model_train)
        reps = extract_mean_token_representations_from_model(
            model=model,
            sequences=dataset.sequences,
            layer=cfg.model_layer,
            vocab_size=world.vocab_size,
            device=cfg.model_train.device,
            batch_size=128,
        )
        rep_report = representation_symmetry_report(
            representations=reps,
            distances=dist,
            pca_rank=min(cfg.decoding_max_rank, reps.shape[0]),
            num_bins=cfg.symmetry_bins,
        )

        # Decoding latent coordinates from representation PCA features.
        feat = rep_report.bundle.pca_coords
        max_rank = min(cfg.decoding_max_rank, feat.shape[1])
        ranks = np.arange(1, max_rank + 1)
        decoding = evaluate_linear_decoding_vs_rank(
            features=feat,
            targets=world.concept.coordinates,
            ranks=ranks.tolist(),
            num_trials=cfg.decoding_trials,
            train_size=max(world.vocab_size // 2, 1),
            ridge_alpha=0.0,
            seed=cfg.policy.seed,
        )

        summary["model"] = {
            "train_loss_first": float(losses[0]),
            "train_loss_last": float(losses[-1]),
            "symmetry_r2": float(rep_report.symmetry.r2),
            "symmetry_mse": float(rep_report.symmetry.mse),
            "top_pca_eigvals": rep_report.bundle.pca_eigvals[: min(6, rep_report.bundle.pca_eigvals.shape[0])],
        }
        summary["decoding"] = {
            "ranks": decoding.ranks,
            "train_rel_mse_mean": decoding.train_rel_mse_mean,
            "test_rel_mse_mean": decoding.test_rel_mse_mean,
            "ridge_alpha": decoding.ridge_alpha,
            "num_trials": decoding.num_trials,
        }

    if cfg.output_dir:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "workflow_summary.json"
        out_path.write_text(json.dumps(_json_compatible(summary), indent=2), encoding="utf-8")
        summary["output_path"] = str(out_path)

    return summary
