"""
Asynchronous job manager for Streamlit interpretability runs.

This module provides a lightweight background queue that can execute B3/B4
launches without blocking the UI thread.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .runs import launch_run


JobStatus = str


@dataclass
class JobRecord:
    """In-memory representation of one background run job."""

    job_id: str
    mode: str
    out_root: str
    run_name: Optional[str]
    exp_config: Optional[str]
    status: JobStatus = "queued"
    submit_ts: float = field(default_factory=time.time)
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    elapsed_s: Optional[float] = None
    run_dir: str = ""
    error: str = ""
    cancel_requested: bool = False
    output: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = field(default_factory=list)

    def add_event(self, message: str) -> None:
        self.events.append({"ts": time.time(), "message": message})


class JobManager:
    """
    Thread-based background queue for interpretability runs.

    Notes:
    - Queued jobs can be cancelled immediately.
    - Running jobs cannot be force-killed safely in this thread model; cancel
      marks the job as "cancel_requested" and records it in logs.
    """

    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="interp-job")
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._futures: Dict[str, Future] = {}

    def submit(
        self,
        mode: str,
        out_root: str,
        run_name: Optional[str] = None,
        exp_config: Optional[str] = None,
    ) -> str:
        """Submit a new background job and return its ID."""
        job_id = uuid.uuid4().hex[:12]
        rec = JobRecord(
            job_id=job_id,
            mode=mode,
            out_root=out_root,
            run_name=run_name,
            exp_config=exp_config,
        )
        rec.add_event("queued")

        with self._lock:
            self._jobs[job_id] = rec
            fut = self._executor.submit(self._run_job, job_id)
            self._futures[job_id] = fut

        return job_id

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            rec = self._jobs[job_id]
            if rec.cancel_requested:
                rec.status = "cancelled"
                rec.end_ts = time.time()
                rec.elapsed_s = rec.end_ts - rec.submit_ts
                rec.add_event("cancelled before start")
                return
            rec.status = "running"
            rec.start_ts = time.time()
            rec.add_event("started")

        try:
            out = launch_run(
                mode=rec.mode,
                out_root=rec.out_root,
                run_name=rec.run_name,
                exp_config=rec.exp_config,
            )
            with self._lock:
                rec.output = out
                rec.run_dir = str(out.get("run_dir", ""))
                rec.end_ts = time.time()
                rec.elapsed_s = rec.end_ts - (rec.start_ts or rec.submit_ts)
                if rec.cancel_requested:
                    rec.status = "cancel_requested"
                    rec.add_event("finished but cancel was requested during execution")
                else:
                    rec.status = "completed"
                    rec.add_event("completed")
        except Exception as exc:  # pragma: no cover - hard to force all branches in tests
            with self._lock:
                rec.end_ts = time.time()
                rec.elapsed_s = rec.end_ts - (rec.start_ts or rec.submit_ts)
                rec.status = "failed"
                rec.error = str(exc)
                rec.add_event(f"failed: {exc}")

    def cancel(self, job_id: str) -> bool:
        """
        Request cancellation for a job.

        Returns True if cancellation request was accepted.
        """
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return False
            fut = self._futures.get(job_id)
            rec.cancel_requested = True
            rec.add_event("cancel requested")
            if fut is not None and fut.cancel():
                rec.status = "cancelled"
                rec.end_ts = time.time()
                rec.elapsed_s = rec.end_ts - rec.submit_ts
                rec.add_event("cancelled while queued")
            return True

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get one job snapshot as a plain dict."""
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            return {
                "job_id": rec.job_id,
                "mode": rec.mode,
                "status": rec.status,
                "submit_ts": rec.submit_ts,
                "start_ts": rec.start_ts,
                "end_ts": rec.end_ts,
                "elapsed_s": rec.elapsed_s,
                "run_dir": rec.run_dir,
                "error": rec.error,
                "cancel_requested": rec.cancel_requested,
                "events": list(rec.events),
                "run_name": rec.run_name,
                "exp_config": rec.exp_config,
            }

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List jobs ordered by submit time descending."""
        with self._lock:
            rows = [
                {
                    "job_id": rec.job_id,
                    "mode": rec.mode,
                    "status": rec.status,
                    "submit_ts": rec.submit_ts,
                    "elapsed_s": rec.elapsed_s,
                    "run_dir": rec.run_dir,
                    "cancel_requested": rec.cancel_requested,
                    "error": rec.error,
                }
                for rec in self._jobs.values()
            ]
        rows.sort(key=lambda x: float(x.get("submit_ts", 0.0)), reverse=True)
        return rows[: max(1, limit)]

    def prune(self, keep_last: int = 200) -> int:
        """Remove old completed/failed/cancelled jobs from memory."""
        with self._lock:
            ordered = sorted(self._jobs.values(), key=lambda r: r.submit_ts, reverse=True)
            keep = {r.job_id for r in ordered[: max(1, keep_last)]}
            removed = 0
            for job_id in list(self._jobs.keys()):
                if job_id in keep:
                    continue
                rec = self._jobs[job_id]
                if rec.status in {"completed", "failed", "cancelled", "cancel_requested"}:
                    self._jobs.pop(job_id, None)
                    self._futures.pop(job_id, None)
                    removed += 1
            return removed
