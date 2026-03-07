from pathlib import Path

from boardGPT.interpretability.reproducibility import (
    collect_environment_snapshot,
    set_global_determinism,
    write_reproducibility_pack,
)


def test_reproducibility_pack_writes_files(tmp_path: Path):
    seed_info = set_global_determinism(seed=7, deterministic=True)
    env = collect_environment_snapshot(include_pip_freeze=False)
    out = write_reproducibility_pack(
        run_dir=str(tmp_path),
        config={"experiment": {"seed": 7, "reproducibility": {"deterministic": True}}},
        seed_info=seed_info,
        env_snapshot=env,
    )

    assert Path(out["reproducibility_json"]).exists()
    assert Path(out["reproducibility_report_md"]).exists()
