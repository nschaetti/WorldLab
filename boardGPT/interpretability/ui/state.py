"""
Streamlit session-state management for interpretability lab.
"""

from typing import Any, Dict

import streamlit as st


DEFAULT_STATE: Dict[str, Any] = {
    "model": None,
    "model_ckpt_path": "",
    "model_cfg_path": "",
    "probe": None,
    "probe_path": "",
    "probe_type": "linear",
    "device": "cpu",
    "input_mode": "token_ids",
    "input_text": "0, 19, 20, 27",
    "tokens_tensor": None,
    "inference_output": None,
    "selected_layer": 0,
    "selected_position": "last",
    "residual_vector": None,
    "probe_output": None,
    "patch_target_cell": 5,
    "patch_target_class": 2,
    "patch_steps": 30,
    "patch_lr": 0.08,
    "patch_reg_strength": 1e-4,
    "patch_clamp_norm": 0.0,
    "patch_output": None,
    "runs_root": "out_interp",
    "run_mode": "b3_demo",
    "run_name": "",
    "run_exp_config": "config/interpretability/b4_example.yaml",
    "recent_runs": [],
    "last_run_output": None,
    "selected_run_a": "",
    "selected_run_b": "",
    "last_error": "",
    "last_success": "",
}


def init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_error(message: str) -> None:
    st.session_state["last_error"] = message
    st.session_state["last_success"] = ""


def set_success(message: str) -> None:
    st.session_state["last_success"] = message
    st.session_state["last_error"] = ""


def reset_input_outputs() -> None:
    st.session_state["tokens_tensor"] = None
    st.session_state["inference_output"] = None
    st.session_state["residual_vector"] = None
    st.session_state["probe_output"] = None
    st.session_state["patch_output"] = None
