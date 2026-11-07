

import torch
from worldlab.nn.board import GPTAEConfig, GPTAE
from worldlab.nn import GameAutoEncoder


# GPT auto-encodeur
config = GPTAEConfig()
model = GameAutoEncoder(config)

# A random sequence
batch_size = 2
seq_len = config.block_size
x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

print("Input shape:", x.shape)             # (b, 60)

# Put a sequence in a model
logits, loss = model(idx=x, targets=x)

# Show input and output shapes
print("logits shape:", logits.shape)
print("loss shape:", loss)

