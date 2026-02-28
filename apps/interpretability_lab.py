#!/usr/bin/env python3
"""
B5-A Streamlit interpretability lab.

Run:
    streamlit run apps/interpretability_lab.py
"""

from __future__ import annotations

import streamlit as st
import torch

from boardGPT.interpretability.ui.activations import get_residual_vector, summarize_vector
from boardGPT.interpretability.ui.compare import compare_run_dirs
from boardGPT.interpretability.ui.inference import run_next_token_inference
from boardGPT.interpretability.ui.loaders import (
    build_untrained_probe,
    load_model_from_paths,
    load_probe_from_path,
)
from boardGPT.interpretability.ui.run_index import discover_runs, read_run_text_preview
from boardGPT.interpretability.ui.runs import launch_run
from boardGPT.interpretability.ui.parsing import parse_input_to_tensor, parse_position
from boardGPT.interpretability.ui.patching import run_activation_patch_experiment
from boardGPT.interpretability.ui.probe_view import (
    format_board_confidence,
    format_board_symbols,
    run_probe_on_vector,
)
from boardGPT.interpretability.ui.state import init_state, reset_input_outputs, set_error, set_success


@st.cache_resource(show_spinner=False)
def _cached_load_model(model_path: str, config_path: str, device: str):
    return load_model_from_paths(model_path=model_path, config_path=config_path, device=device)


@st.cache_resource(show_spinner=False)
def _cached_load_probe(path: str, input_dim: int, probe_type: str, hidden_dim: int, device: str):
    return load_probe_from_path(
        probe_path=path,
        input_dim=input_dim,
        probe_type=probe_type,
        hidden_dim=hidden_dim,
        device=device,
    )


def _sidebar_model_probe() -> None:
    st.sidebar.header("Model / Probe")

    st.session_state["model_ckpt_path"] = st.sidebar.text_input(
        "Model checkpoint (.pt/.pth/.safetensors)",
        value=st.session_state["model_ckpt_path"] or "out/ckpt_iter240000.pt",
    )
    st.session_state["model_cfg_path"] = st.sidebar.text_input(
        "Model config (.yaml/.json)",
        value=st.session_state["model_cfg_path"] or "config/config.yaml",
    )
    st.session_state["device"] = st.sidebar.selectbox(
        "Device",
        options=["cpu", "cuda"],
        index=0 if st.session_state["device"] == "cpu" else 1,
    )

    if st.sidebar.button("Load Model", use_container_width=True):
        try:
            model = _cached_load_model(
                st.session_state["model_ckpt_path"],
                st.session_state["model_cfg_path"],
                st.session_state["device"],
            )
            st.session_state["model"] = model
            st.session_state["selected_layer"] = 0
            reset_input_outputs()
            set_success("Model loaded successfully")
        except Exception as exc:
            set_error(f"Load model failed: {exc}")

    st.sidebar.divider()

    st.session_state["probe_type"] = st.sidebar.selectbox(
        "Probe type",
        options=["linear", "mlp"],
        index=0 if st.session_state["probe_type"] == "linear" else 1,
    )
    probe_hidden = st.sidebar.number_input("Probe hidden dim (mlp)", min_value=16, value=256, step=16)
    st.session_state["probe_path"] = st.sidebar.text_input(
        "Probe checkpoint (.pt, optional)",
        value=st.session_state["probe_path"],
        placeholder="Leave empty to create an untrained probe",
    )

    if st.sidebar.button("Load Probe", use_container_width=True):
        model = st.session_state.get("model")
        if model is None:
            set_error("Load a model before loading a probe")
            return
        try:
            if st.session_state["probe_path"].strip():
                probe = _cached_load_probe(
                    st.session_state["probe_path"],
                    input_dim=model.config.n_embd,
                    probe_type=st.session_state["probe_type"],
                    hidden_dim=int(probe_hidden),
                    device=st.session_state["device"],
                )
            else:
                probe = build_untrained_probe(
                    input_dim=model.config.n_embd,
                    probe_type=st.session_state["probe_type"],
                    hidden_dim=int(probe_hidden),
                ).to(st.session_state["device"])
                probe.eval()
            st.session_state["probe"] = probe
            st.session_state["probe_output"] = None
            set_success("Probe ready")
        except Exception as exc:
            set_error(f"Load probe failed: {exc}")


def _messages() -> None:
    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])
    elif st.session_state.get("last_success"):
        st.success(st.session_state["last_success"])


def _tab_model_probe() -> None:
    st.subheader("Model & Probe")
    model = st.session_state.get("model")
    probe = st.session_state.get("probe")

    if model is None:
        st.info("No model loaded yet. Use the sidebar to load a model.")
    else:
        st.write(
            {
                "checkpoint": st.session_state["model_ckpt_path"],
                "config": st.session_state["model_cfg_path"],
                "device": st.session_state["device"],
                "n_layer": model.config.n_layer,
                "n_embd": model.config.n_embd,
                "block_size": model.config.block_size,
                "vocab_size": model.config.vocab_size,
            }
        )

    if probe is None:
        st.info("No probe loaded. You can still use inference and activations.")
    else:
        st.write({"probe_type": st.session_state["probe_type"], "probe_path": st.session_state["probe_path"]})


def _tab_input_prediction() -> None:
    st.subheader("Input & Prediction")
    model = st.session_state.get("model")

    mode = st.radio("Input mode", options=["token_ids", "moves"], horizontal=True, index=0)
    st.session_state["input_mode"] = mode
    default_text = "0, 19, 20, 27" if mode == "token_ids" else "d3,c3,c4,e3"
    st.session_state["input_text"] = st.text_area("Sequence", value=st.session_state.get("input_text") or default_text, height=120)

    if st.button("Parse Input", use_container_width=True):
        if model is None:
            set_error("Load a model first")
            return
        try:
            tokens = parse_input_to_tensor(
                text=st.session_state["input_text"],
                mode=st.session_state["input_mode"],
                block_size=model.config.block_size,
                device=st.session_state["device"],
            )
            st.session_state["tokens_tensor"] = tokens
            st.session_state["inference_output"] = None
            st.session_state["residual_vector"] = None
            set_success("Input parsed")
        except Exception as exc:
            set_error(f"Parse failed: {exc}")

    tokens = st.session_state.get("tokens_tensor")
    if torch.is_tensor(tokens):
        st.write(f"Token tensor shape: {tuple(tokens.shape)}")
        st.code(tokens[0].detach().cpu().tolist())

    if st.button("Run Inference", use_container_width=True):
        if model is None:
            set_error("Load a model first")
            return
        if not torch.is_tensor(tokens):
            set_error("Parse input first")
            return
        try:
            output = run_next_token_inference(model, tokens=tokens, top_k=10)
            st.session_state["inference_output"] = output
            set_success("Inference done")
        except Exception as exc:
            set_error(f"Inference failed: {exc}")

    inference_output = st.session_state.get("inference_output")
    if isinstance(inference_output, dict):
        st.write(
            {
                "prediction_token_id": inference_output["prediction_token_id"],
                "prediction_prob": inference_output["prediction_prob"],
            }
        )
        st.dataframe(inference_output["top_k"], use_container_width=True)


def _tab_activations() -> None:
    st.subheader("Activations")
    model = st.session_state.get("model")
    tokens = st.session_state.get("tokens_tensor")

    if model is None:
        st.info("Load a model first")
        return

    st.session_state["selected_layer"] = st.slider(
        "Layer",
        min_value=0,
        max_value=max(0, model.config.n_layer - 1),
        value=min(st.session_state["selected_layer"], model.config.n_layer - 1),
        step=1,
    )
    pos_text = st.text_input("Position (last or index)", value=str(st.session_state.get("selected_position", "last")))
    st.session_state["selected_position"] = pos_text

    if st.button("Extract Activations", use_container_width=True):
        if not torch.is_tensor(tokens):
            set_error("Parse input first")
            return
        try:
            pos_idx = parse_position(pos_text, seq_len=tokens.size(1))
            vec = get_residual_vector(
                model=model,
                tokens=tokens,
                layer=int(st.session_state["selected_layer"]),
                position=pos_idx,
                device=st.session_state["device"],
            )
            st.session_state["residual_vector"] = vec
            st.session_state["selected_position"] = str(pos_idx)
            st.session_state["probe_output"] = None
            set_success("Activation extracted")
        except Exception as exc:
            set_error(f"Activation extraction failed: {exc}")

    vec = st.session_state.get("residual_vector")
    if torch.is_tensor(vec):
        stats = summarize_vector(vec)
        st.write(stats)
        st.line_chart(vec.numpy())


def _tab_probe_dashboard() -> None:
    st.subheader("Probe Dashboard")
    probe = st.session_state.get("probe")
    vec = st.session_state.get("residual_vector")

    if probe is None:
        st.info("No probe loaded. Load probe from sidebar.")
        return
    if not torch.is_tensor(vec):
        st.info("Extract activations first.")
        return

    if st.button("Run Probe", use_container_width=True):
        try:
            out = run_probe_on_vector(probe=probe, vector=vec, device=st.session_state["device"])
            st.session_state["probe_output"] = out
            set_success("Probe inference done")
        except Exception as exc:
            set_error(f"Probe run failed: {exc}")

    out = st.session_state.get("probe_output")
    if isinstance(out, dict):
        pred = out["pred"]
        conf = out["conf"]
        board = format_board_symbols(pred)
        conf_grid = format_board_confidence(conf)
        st.write("Predicted board (. empty, W white, B black)")
        st.table(board)
        st.write("Confidence map")
        st.table([[round(x, 3) for x in row] for row in conf_grid])


def _tab_activation_patching() -> None:
    st.subheader("Activation Patching (B5-B)")
    model = st.session_state.get("model")
    probe = st.session_state.get("probe")
    tokens = st.session_state.get("tokens_tensor")
    vec = st.session_state.get("residual_vector")

    if model is None or probe is None:
        st.info("Load both model and probe to run patching.")
        return
    if not torch.is_tensor(tokens):
        st.info("Parse input first.")
        return
    if not torch.is_tensor(vec):
        st.info("Extract activations first.")
        return

    c1, c2, c3 = st.columns(3)
    st.session_state["patch_target_cell"] = c1.number_input(
        "Target cell", min_value=0, max_value=63, value=int(st.session_state["patch_target_cell"]), step=1
    )
    st.session_state["patch_target_class"] = c2.number_input(
        "Target class", min_value=0, max_value=2, value=int(st.session_state["patch_target_class"]), step=1
    )
    st.session_state["patch_steps"] = c3.number_input(
        "Steps", min_value=1, max_value=1000, value=int(st.session_state["patch_steps"]), step=1
    )

    c4, c5, c6 = st.columns(3)
    st.session_state["patch_lr"] = c4.number_input(
        "Learning rate", min_value=1e-4, max_value=1.0, value=float(st.session_state["patch_lr"]), step=0.01, format="%.4f"
    )
    st.session_state["patch_reg_strength"] = c5.number_input(
        "Reg strength", min_value=0.0, max_value=1.0, value=float(st.session_state["patch_reg_strength"]), step=1e-4, format="%.6f"
    )
    st.session_state["patch_clamp_norm"] = c6.number_input(
        "Clamp norm (0 = off)", min_value=0.0, max_value=1000.0, value=float(st.session_state["patch_clamp_norm"]), step=0.1
    )

    if st.button("Run Activation Patch", use_container_width=True):
        try:
            pos_idx = parse_position(str(st.session_state["selected_position"]), seq_len=tokens.size(1))
            clamp = float(st.session_state["patch_clamp_norm"])
            clamp_opt = None if clamp <= 0 else clamp

            out = run_activation_patch_experiment(
                model=model,
                probe=probe,
                tokens=tokens,
                layer=int(st.session_state["selected_layer"]),
                position=pos_idx,
                target_cell=int(st.session_state["patch_target_cell"]),
                target_class=int(st.session_state["patch_target_class"]),
                steps=int(st.session_state["patch_steps"]),
                lr=float(st.session_state["patch_lr"]),
                reg_strength=float(st.session_state["patch_reg_strength"]),
                clamp_norm=clamp_opt,
                device=st.session_state["device"],
            )
            st.session_state["patch_output"] = out
            set_success("Activation patch run completed")
        except Exception as exc:
            set_error(f"Activation patch failed: {exc}")

    patch = st.session_state.get("patch_output")
    if isinstance(patch, dict):
        intervention = patch.get("intervention", {})
        before_inf = patch.get("before_inference", {})
        after_inf = patch.get("after_inference", {})

        st.write(
            {
                "target_prob_before": intervention.get("before_target_prob"),
                "target_prob_after": intervention.get("after_target_prob"),
                "target_prob_gain": intervention.get("target_prob_gain"),
                "target_logit_gain": intervention.get("target_logit_gain"),
                "delta_norm": intervention.get("delta_norm"),
                "success": intervention.get("success"),
            }
        )

        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("**Model prediction - Before**")
            st.write(
                {
                    "token_id": before_inf.get("prediction_token_id"),
                    "prob": before_inf.get("prediction_prob"),
                }
            )
            st.dataframe(before_inf.get("top_k", []), use_container_width=True)
        with col_after:
            st.markdown("**Model prediction - After**")
            st.write(
                {
                    "token_id": after_inf.get("prediction_token_id"),
                    "prob": after_inf.get("prediction_prob"),
                }
            )
            st.dataframe(after_inf.get("top_k", []), use_container_width=True)

        before_probs = intervention.get("before_board_probs")
        after_probs = intervention.get("after_board_probs")
        if torch.is_tensor(before_probs) and torch.is_tensor(after_probs):
            before_pred = torch.argmax(before_probs, dim=-1)
            before_conf = torch.max(before_probs, dim=-1).values
            after_pred = torch.argmax(after_probs, dim=-1)
            after_conf = torch.max(after_probs, dim=-1).values

            bcol, acol = st.columns(2)
            with bcol:
                st.markdown("**Probe board - Before**")
                st.table(format_board_symbols(before_pred))
                st.table([[round(x, 3) for x in row] for row in format_board_confidence(before_conf)])
            with acol:
                st.markdown("**Probe board - After**")
                st.table(format_board_symbols(after_pred))
                st.table([[round(x, 3) for x in row] for row in format_board_confidence(after_conf)])


def _tab_runs_compare() -> None:
    st.subheader("Runs & Compare (B5-C)")
    sub1, sub2, sub3 = st.tabs(["Launcher", "Explorer", "Compare"])

    with sub1:
        st.markdown("Launch B3/B4 runs directly from the dashboard")
        c1, c2 = st.columns(2)
        st.session_state["run_mode"] = c1.selectbox(
            "Run mode",
            options=["b3_demo", "b4_real", "b4_batch"],
            index=["b3_demo", "b4_real", "b4_batch"].index(st.session_state.get("run_mode", "b3_demo")),
        )
        st.session_state["run_name"] = c2.text_input("Run name (optional)", value=st.session_state.get("run_name", ""))

        c3, c4 = st.columns(2)
        st.session_state["runs_root"] = c3.text_input("Runs root", value=st.session_state.get("runs_root", "out_interp"))
        st.session_state["run_exp_config"] = c4.text_input(
            "B4 exp config",
            value=st.session_state.get("run_exp_config", "config/interpretability/b4_example.yaml"),
        )

        if st.button("Launch Run", use_container_width=True):
            try:
                output = launch_run(
                    mode=st.session_state["run_mode"],
                    out_root=st.session_state["runs_root"],
                    run_name=(st.session_state["run_name"].strip() or None),
                    exp_config=st.session_state["run_exp_config"],
                )
                st.session_state["last_run_output"] = output
                rec = st.session_state.get("recent_runs", [])
                elapsed_val = output.get("elapsed_s", 0.0)
                elapsed = float(elapsed_val) if isinstance(elapsed_val, (int, float)) else 0.0
                rec.insert(0, {"mode": output.get("mode"), "run_dir": output.get("run_dir"), "elapsed_s": round(elapsed, 3)})
                st.session_state["recent_runs"] = rec[:20]
                set_success(f"Run completed in {elapsed:.2f}s: {output.get('run_dir')}")
            except Exception as exc:
                set_error(f"Run launch failed: {exc}")

        last = st.session_state.get("last_run_output")
        if isinstance(last, dict):
            st.write({"mode": last.get("mode"), "run_dir": last.get("run_dir"), "elapsed_s": last.get("elapsed_s")})

        recent = st.session_state.get("recent_runs", [])
        if isinstance(recent, list) and recent:
            st.markdown("**Recent runs**")
            st.dataframe(recent, use_container_width=True)

    with sub2:
        st.markdown("Explore run artifacts from disk")
        root = st.text_input("Scan root", value=st.session_state.get("runs_root", "out_interp"), key="scan_root_input")
        runs = discover_runs(root)
        if not runs:
            st.info("No runs found")
        else:
            st.write(f"Found {len(runs)} runs")
            st.dataframe(runs, use_container_width=True)

            options = [r["run_dir"] for r in runs]
            selected = st.selectbox("Select run", options=options, key="explorer_selected_run")
            st.write({"run_dir": selected})

            report_preview = read_run_text_preview(selected, "report.md", max_chars=4000)
            if report_preview:
                with st.expander("report.md preview", expanded=False):
                    st.markdown(report_preview)

            summary_preview = read_run_text_preview(selected, "summary.json", max_chars=4000)
            if summary_preview:
                with st.expander("summary.json preview", expanded=False):
                    st.code(summary_preview, language="json")

    with sub3:
        st.markdown("Compare two runs (A vs B)")
        root = st.text_input("Scan root for compare", value=st.session_state.get("runs_root", "out_interp"), key="compare_root_input")
        runs = discover_runs(root)
        options = [r["run_dir"] for r in runs]
        if len(options) < 2:
            st.info("Need at least two runs to compare")
        else:
            default_a = options[0]
            default_b = options[1]
            run_a = st.selectbox("Run A", options=options, index=0, key="compare_run_a")
            run_b = st.selectbox("Run B", options=options, index=1, key="compare_run_b")

            if st.button("Compare Runs", use_container_width=True):
                try:
                    cmp_out = compare_run_dirs(run_a, run_b)
                    st.session_state["compare_output"] = cmp_out
                    set_success("Comparison ready")
                except Exception as exc:
                    set_error(f"Comparison failed: {exc}")

            cmp_out = st.session_state.get("compare_output")
            if isinstance(cmp_out, dict):
                st.write({"run_a": cmp_out.get("run_a"), "run_b": cmp_out.get("run_b")})
                st.markdown("**Global deltas**")
                st.write(cmp_out.get("global_delta", {}))
                st.markdown("**Layer deltas**")
                st.dataframe(cmp_out.get("layer_deltas", []), use_container_width=True)
                st.markdown("**Top contributor overlap**")
                st.write({"overlap_count": cmp_out.get("top_overlap_count", 0)})
                st.dataframe(cmp_out.get("top_overlap", []), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="BoardGPT Interpretability Lab", layout="wide")
    st.title("BoardGPT Interpretability Lab (B5-C)")

    init_state()
    _sidebar_model_probe()
    _messages()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Model & Probe",
        "Input & Prediction",
        "Activations",
        "Probe Dashboard",
        "Activation Patching",
        "Runs & Compare",
    ])
    with tab1:
        _tab_model_probe()
    with tab2:
        _tab_input_prediction()
    with tab3:
        _tab_activations()
    with tab4:
        _tab_probe_dashboard()
    with tab5:
        _tab_activation_patching()
    with tab6:
        _tab_runs_compare()


if __name__ == "__main__":
    main()
