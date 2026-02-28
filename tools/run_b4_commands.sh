#!/usr/bin/env bash
set -euo pipefail

# B4 real-mode commands (inference only: no training)

# 1) Single real metrics run
python tools/interpretability_cli.py real-metrics \
  --exp-config config/interpretability/b4_example.yaml \
  --run-name b4_real_metrics

# 2) Single real attribution run
python tools/interpretability_cli.py real-attrib \
  --exp-config config/interpretability/b4_example.yaml \
  --run-name b4_real_attrib

# 3) Batch comparison (multi-seed, multi-checkpoint)
python tools/interpretability_cli.py real-batch \
  --exp-config config/interpretability/b4_example.yaml \
  --run-name b4_real_batch

# 4) Same commands via games CLI bridge (optional)
python -m boardGPT.games.cli othello interp real-metrics \
  --exp-config config/interpretability/b4_example.yaml \
  --run-name b4_real_metrics_bridge

python -m boardGPT.games.cli othello interp real-batch \
  --exp-config config/interpretability/b4_example.yaml \
  --run-name b4_real_batch_bridge
