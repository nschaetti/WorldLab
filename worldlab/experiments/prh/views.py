"""Build positives within one modality, with or without shared scene identity."""

import torch


class TextViews:
    def __init__(self, tokenizer, length=64, mask=0.15):
        self.cls = tokenizer.cls_token_id
        self.sep = tokenizer.sep_token_id
        self.mask = tokenizer.mask_token_id
        self.pad = tokenizer.pad_token_id
        if any(value is None for value in (self.cls, self.sep, self.mask, self.pad)):
            raise ValueError("The tokenizer must provide CLS, SEP, MASK and PAD")
        if length < 4 or not 0 <= mask <= 1:
            raise ValueError("Expected sequence length >= 4 and a mask fraction in [0,1]")
        self.length = length
        self.mask_probability = mask

    def _pack(self, parts, length):
        tokens = [self.cls]
        for part in parts:
            tokens.extend(part.tolist())
            tokens.append(self.sep)
        return torch.tensor(tokens + [self.pad] * (length - len(tokens)), dtype=torch.long)

    def _span(self, tokens, scale):
        tokens = tokens[:self.length - 2]
        if len(tokens):
            lo, hi = scale
            keep = max(1, round(len(tokens) * (lo + (hi - lo) * torch.rand(()).item())))
            start = torch.randint(len(tokens) - keep + 1, ()).item()
            tokens = tokens[start:start + keep].clone()
            tokens[torch.rand(keep) < self.mask_probability] = self.mask
        return self._pack([tokens], self.length)

    def evaluation(self, tokens):
        return self._pack([tokens[:self.length - 2]], self.length)

    def training(self, captions, scales, *, good=False, next_token=False):
        if not captions:
            raise ValueError("A scene needs at least one caption")
        if next_token and good and len(captions) > 1:
            picks = torch.randperm(len(captions))[:2]
            capacity = (2 * self.length - 3) // 2
            return self._pack([captions[index][:capacity] for index in picks], 2 * self.length)
        if next_token:
            tokens = self.evaluation(captions[torch.randint(len(captions), ()).item()])
            return tokens
        source = captions if good else [captions[torch.randint(len(captions), ()).item()]]
        return tuple(self._span(source[torch.randint(len(source), ()).item()], scale) for scale in scales)
