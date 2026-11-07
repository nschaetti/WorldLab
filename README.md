# WorldLab

Research code for world models, from autoregressive board-game transformers to joint-embedding predictive architectures.

- **LeJEPA** lives in `worldlab.nn`. The [reproduction guide](docs/lejepa.md) covers pretraining, linear probes, fine-tuning and experiment matrices.
- **LeWorldModel** adds action-conditioned latent dynamics and planning. See the [reproduction guide](docs/lewm.md) for training, control and physical diagnostics.
- **Platonic Representation Hypothesis** compares independently trained vision and text representations. See the [PRH experiment guide](docs/prh.md).
- **BoardGPT** lives in `worldlab.board`, with its PyTorch models in `worldlab.nn`. See the [training configuration](README_config.md) and [interpretability toolkit](README_interpretability.md).

```bash
python -m pip install -e '.[lejepa]'
worldlab-lejepa train --preset smoke --device cpu --output runs/lejepa-smoke
```

Install the board dependencies with `python -m pip install -e '.[board]'`. Import the board models alongside the JEPAs:

```python
from worldlab.nn import GPTConfig, GameGPT, GameAutoEncoder, LeJEPA, LeWorldModel
from worldlab.board.games import OthelloGame
```

Use `worldlab-board` for the game CLI and `worldlab-board-train`, `worldlab-board-autoencoder` or `worldlab-board-probe` for training. Existing `boardGPT` imports and root training scripts are compatibility entry points. Board model state dicts and training checkpoint keys use the original format.
