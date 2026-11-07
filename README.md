# WorldLab

Research code for world models, from autoregressive board-game transformers to joint-embedding predictive architectures.

- **LeJEPA** lives in `worldlab.nn`. The [reproduction guide](docs/lejepa.md) covers pretraining, linear probes, fine-tuning and experiment matrices.
- **LeWorldModel** adds action-conditioned latent dynamics and planning. See the [reproduction guide](docs/lewm.md) for training, control and physical diagnostics.
- **BoardGPT** lives in `boardGPT`. See the [training configuration](README_config.md) and [interpretability toolkit](README_interpretability.md).

```bash
python -m pip install -e '.[lejepa]'
worldlab-lejepa train --preset smoke --device cpu --output runs/lejepa-smoke
```
