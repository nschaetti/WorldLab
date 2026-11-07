"""One record per scene. Training reads only the active modality."""

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from worldlab.experiments.lejepa.data import _augment, evaluation_transform

from .views import TextViews


def load_scenes(path, root=None):
    """Read a scene JSONL manifest or the upstream Flickr30k Karpathy CSV."""
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "flickr_annotations_30k.csv"
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [
                {"id": row["img_id"], "split": row["split"], "captions": json.loads(row["raw"]),
                 "images": [str(Path("flickr30k-images") / row["filename"])]}
                for row in csv.DictReader(stream)
            ]
    else:
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("The scene manifest is empty")
    identifiers = [str(row["id"]) for row in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError("Scene IDs must be unique, including across data splits")
    root = Path(root).expanduser() if root else path.parent
    return rows, root


class SceneViews(Dataset):
    def __init__(self, rows, root, config, tokenizer=None, *, training=False):
        self.rows, self.root, self.config, self.training = rows, root, config, training
        self.tokenizer = tokenizer
        self.text_views = None
        if config.modality == "text":
            self.text_views = TextViews(tokenizer, config.max_len, config.mask)
        else:
            self.global_crop = _augment(config.resolution, (0.3, 1.0), "paper")
            self.local_crop = _augment(config.local_resolution, (0.05, 0.3), "paper")
            self.eval_crop = evaluation_transform(config.resolution, "paper")

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _draw(items):
        return items[torch.randint(len(items), ()).item()]

    def _image(self, name):
        with Image.open(self.root / name) as image:
            return image.convert("RGB")

    def _text(self, captions, good):
        if not isinstance(captions, list) or not captions or not all(isinstance(caption, str) for caption in captions):
            raise ValueError("Each text scene needs a nonempty list of captions")
        cfg = self.config
        if not self.training:
            captions = captions[:1]
        ids = self.tokenizer(captions, add_special_tokens=False, truncation=True,
                             max_length=cfg.max_len)["input_ids"]
        tokens = [torch.as_tensor(part, dtype=torch.long) for part in ids]
        if not self.training:
            return self.text_views.evaluation(tokens[0])
        scales = (cfg.global_scale,) * cfg.global_views + (cfg.local_scale,) * cfg.local_views
        result = self.text_views.training(tokens, scales, good=good, next_token=cfg.objective == "ntp")
        if cfg.objective == "ntp" and cfg.good_views:
            result = F.pad(result, (0, 2 * cfg.max_len - len(result)), value=self.text_views.pad)
        return result

    def _visual(self, names, good):
        if not isinstance(names, list) or not names:
            raise ValueError("Each visual scene needs a nonempty list of image paths")
        cfg = self.config
        if not self.training:
            index = len(names) // 2
            return self.eval_crop(self._image(names[index]))
        # For bad views, sample the photograph once, before making any crops.
        shared = None if good else self._image(self._draw(names))
        crops = (self.global_crop,) * cfg.global_views + (self.local_crop,) * cfg.local_views
        return tuple(crop(self._image(self._draw(names)) if shared is None else shared) for crop in crops)

    def __getitem__(self, index):
        row = self.rows[index]
        good = self.training and self.config.good_views > 0 and torch.rand(()).item() < self.config.good_views
        values = (self._text(row.get("captions"), good) if self.text_views
                  else self._visual(row.get("images"), good))
        if self.training:
            return values
        return values, str(row["id"]), str(row.get("split", "test"))
