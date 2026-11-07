"""
Analysis utilities for symmetry-in-statistics experiments.

This package contains the first building blocks for reproducing and extending
the analysis pipeline from:

"Symmetry in language statistics shapes the geometry of model representations"
(Karkada et al., 2026)

The Phase 1 scope focuses on:
1) weighted co-occurrence statistics,
2) latent concept definitions and distances,
3) synthetic sequence generation from explicit probability policies.
"""

from .cooccurrence import CooccurrenceStats, build_cooccurrence_statistics
from .concepts import (
    ConceptSpec,
    lattice_1d_concept,
    lattice_nd_concept,
    pairwise_distances,
)
from .synthetic_sequences import (
    PolicyConfig,
    SyntheticSequenceDataset,
    SyntheticWorld,
    generate_synthetic_sequences,
)
from .theory_karkada import (
    BinnedProfile,
    ExponentialFitResult,
    SymmetryDiagnostic,
    build_exponential_kernel_matrix,
    distance_binned_profile,
    fit_exponential_kernel_from_matrix,
    fit_exponential_profile,
    pca_from_gram,
    translation_symmetry_diagnostic,
)
from .representations import (
    RepresentationBundle,
    RepresentationSymmetryReport,
    build_representation_bundle,
    compute_gram_matrix,
    extract_mean_token_representations_from_model,
    representation_symmetry_report,
)
from .synthetic_gpt import (
    SyntheticGPTTrainConfig,
    build_tiny_gpt_for_synthetic,
    train_gpt_on_synthetic_sequences,
)
from .decoding import (
    DecodingCurveResult,
    evaluate_linear_decoding_vs_rank,
    fit_linear_decoder,
    mse,
    predict_linear_decoder,
    relative_mse,
)
from .ablations import (
    HelperCurveResult,
    ReconstructionMetrics,
    ablate_target_block,
    compute_reconstruction_metrics,
    helper_reconstruction_curve,
    procrustes_error,
    reconstruct_target_with_helpers,
    select_helpers_by_coupling,
)
from .workflows import WorkflowConfig, run_synthetic_symmetry_workflow
from .benchmark import BenchmarkRunRecord, flatten_workflow_summary, run_workflow_benchmark


__all__ = [
    "CooccurrenceStats",
    "build_cooccurrence_statistics",
    "ConceptSpec",
    "lattice_1d_concept",
    "lattice_nd_concept",
    "pairwise_distances",
    "PolicyConfig",
    "SyntheticSequenceDataset",
    "SyntheticWorld",
    "generate_synthetic_sequences",
    "BinnedProfile",
    "ExponentialFitResult",
    "SymmetryDiagnostic",
    "distance_binned_profile",
    "translation_symmetry_diagnostic",
    "fit_exponential_profile",
    "fit_exponential_kernel_from_matrix",
    "build_exponential_kernel_matrix",
    "pca_from_gram",
    "RepresentationBundle",
    "RepresentationSymmetryReport",
    "compute_gram_matrix",
    "build_representation_bundle",
    "extract_mean_token_representations_from_model",
    "representation_symmetry_report",
    "SyntheticGPTTrainConfig",
    "build_tiny_gpt_for_synthetic",
    "train_gpt_on_synthetic_sequences",
    "DecodingCurveResult",
    "fit_linear_decoder",
    "predict_linear_decoder",
    "mse",
    "relative_mse",
    "evaluate_linear_decoding_vs_rank",
    "ReconstructionMetrics",
    "HelperCurveResult",
    "ablate_target_block",
    "select_helpers_by_coupling",
    "procrustes_error",
    "compute_reconstruction_metrics",
    "reconstruct_target_with_helpers",
    "helper_reconstruction_curve",
    "WorkflowConfig",
    "run_synthetic_symmetry_workflow",
    "BenchmarkRunRecord",
    "flatten_workflow_summary",
    "run_workflow_benchmark",
]
