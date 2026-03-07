from pathlib import Path
import json

from boardGPT.interpretability.ui.run_index import discover_runs, parse_tags, read_run_meta, write_run_meta


def test_run_meta_roundtrip(tmp_path: Path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = write_run_meta(str(run_dir), tags=["alpha", "beta"], notes="hello")
    assert Path(meta_path).exists()

    meta = read_run_meta(str(run_dir))
    assert meta.get("tags") == ["alpha", "beta"]
    assert meta.get("notes") == "hello"


def test_discover_runs_includes_tags_and_notes(tmp_path: Path):
    run_dir = tmp_path / "x" / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "summary.json").write_text(json.dumps({"mode": "b4_real"}), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"model": {"model_path": "m.pt"}, "dataset": {"path": "d.pkl"}}),
        encoding="utf-8",
    )
    write_run_meta(str(run_dir), tags=["nightly", "exp"], notes="notes")

    rows = discover_runs(str(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["tags"] == "nightly, exp"
    assert row["notes"] == "notes"


def test_parse_tags_normalizes_values():
    out = parse_tags(" a, b ,, c ")
    assert out == ["a", "b", "c"]
