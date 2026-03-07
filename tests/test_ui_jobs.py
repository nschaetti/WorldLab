import time

import boardGPT.interpretability.ui.jobs as jobs_mod


def test_job_manager_submit_complete_and_cancel(monkeypatch):
    def fake_launch_run(mode, out_root, run_name=None, exp_config=None):
        time.sleep(0.05)
        return {"run_dir": f"{out_root}/dummy", "mode": mode, "elapsed_s": 0.05}

    monkeypatch.setattr(jobs_mod, "launch_run", fake_launch_run)

    mgr = jobs_mod.JobManager(max_workers=1)
    job_id = mgr.submit(mode="b4_real", out_root="/tmp/out", run_name="r1", exp_config="cfg.yaml")

    # Poll until terminal state.
    for _ in range(50):
        row = mgr.get(job_id)
        if row and row.get("status") in {"completed", "failed", "cancelled", "cancel_requested"}:
            break
        time.sleep(0.02)

    row = mgr.get(job_id)
    assert row is not None
    assert row["status"] in {"completed", "cancel_requested"}

    # Canceling a finished job still records request but should not crash.
    assert mgr.cancel(job_id) is True
    row2 = mgr.get(job_id)
    assert row2 is not None
    assert row2["cancel_requested"] is True


def test_job_manager_cancel_queued(monkeypatch):
    # Create a slow launch to keep second job queued.
    def slow_launch_run(mode, out_root, run_name=None, exp_config=None):
        time.sleep(0.2)
        return {"run_dir": f"{out_root}/dummy", "mode": mode, "elapsed_s": 0.2}

    monkeypatch.setattr(jobs_mod, "launch_run", slow_launch_run)

    mgr = jobs_mod.JobManager(max_workers=1)
    j1 = mgr.submit(mode="b4_real", out_root="/tmp/out", run_name="r1", exp_config="cfg.yaml")
    j2 = mgr.submit(mode="b4_real", out_root="/tmp/out", run_name="r2", exp_config="cfg.yaml")

    # Cancel second while queued.
    assert mgr.cancel(j2) is True

    # Wait first to complete.
    for _ in range(60):
        r1 = mgr.get(j1)
        if r1 and r1.get("status") in {"completed", "failed", "cancelled", "cancel_requested"}:
            break
        time.sleep(0.02)

    r2 = mgr.get(j2)
    assert r2 is not None
    assert r2["cancel_requested"] is True
    assert r2["status"] in {"cancelled", "cancel_requested", "queued", "running", "completed"}
