"""
Interpretability utilities for BoardGPT.
"""

from .hooked_model import HookedModel
from .spec import (
    OTHELLO_BOARD_SIZE,
    OTHELLO_NUM_BOARD_CELLS,
    OTHELLO_NUM_MOVE_TOKENS,
    OTHELLO_NUM_VOCAB_TOKENS,
    get_othello_move_mappings,
    move_ids_to_notation,
    notation_to_board_labels,
    move_ids_to_board_labels,
)
from .activation_extractor import extract_residuals, extract_dataset_residuals
from .probes import (
    LinearBoardProbe,
    MLPBoardProbe,
    fit_probe,
    eval_probe,
    predict_board,
)
from .interventions import (
    InterventionTarget,
    InterventionConfig,
    optimize_residual_delta,
    apply_residual_delta,
    run_probe_intervention,
)
from .protocols import (
    ProtocolConfig,
    run_single_target_protocol,
    run_layer_sweep_protocol,
    run_b2_metrics_protocol,
    run_b2_attribution_protocol,
)
from .metrics_intervention import (
    compute_success_rate,
    compute_target_gain_stats,
    compute_delta_norm_stats,
    compute_off_target_impact,
    aggregate_by_layer,
    aggregate_by_cell,
)
from .attribution import (
    compute_intervention_attribution,
    compute_layer_cell_attribution_map,
    rank_top_contributors,
    compute_intervention_attribution_summary,
)
from .io import save_run_artifacts, load_run_artifacts
from .report import export_metrics_csv, export_top_contributors_csv, generate_markdown_report
from .pipeline import build_demo_model_probe, run_b3_demo_pipeline
from .real_pipeline import (
    load_exp_config,
    run_b4_single_from_config,
    run_b4_from_config_file,
    run_b4_batch_from_config_file,
)


__all__ = [
    "HookedModel",
    "OTHELLO_BOARD_SIZE",
    "OTHELLO_NUM_BOARD_CELLS",
    "OTHELLO_NUM_MOVE_TOKENS",
    "OTHELLO_NUM_VOCAB_TOKENS",
    "get_othello_move_mappings",
    "move_ids_to_notation",
    "notation_to_board_labels",
    "move_ids_to_board_labels",
    "extract_residuals",
    "extract_dataset_residuals",
    "LinearBoardProbe",
    "MLPBoardProbe",
    "fit_probe",
    "eval_probe",
    "predict_board",
    "InterventionTarget",
    "InterventionConfig",
    "optimize_residual_delta",
    "apply_residual_delta",
    "run_probe_intervention",
    "ProtocolConfig",
    "run_single_target_protocol",
    "run_layer_sweep_protocol",
    "run_b2_metrics_protocol",
    "run_b2_attribution_protocol",
    "compute_success_rate",
    "compute_target_gain_stats",
    "compute_delta_norm_stats",
    "compute_off_target_impact",
    "aggregate_by_layer",
    "aggregate_by_cell",
    "compute_intervention_attribution",
    "compute_layer_cell_attribution_map",
    "rank_top_contributors",
    "compute_intervention_attribution_summary",
    "save_run_artifacts",
    "load_run_artifacts",
    "export_metrics_csv",
    "export_top_contributors_csv",
    "generate_markdown_report",
    "build_demo_model_probe",
    "run_b3_demo_pipeline",
    "load_exp_config",
    "run_b4_single_from_config",
    "run_b4_from_config_file",
    "run_b4_batch_from_config_file",
]
