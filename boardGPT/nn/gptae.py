"""
Variants of GPT-style autoencoders used for compressing board game sequences.

The module implements configuration helpers, utilities for loading pretrained
weights, and the main ``GPTAE`` model that maps tokenized games to latent
representations and back.
"""

# Imports
import yaml
import torch
import torch.nn as nn
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from . import GPTConfig
from .layer_norm import LayerNorm
from .block import Block


@dataclass
class GPTAEConfig:
    """
    Hyper-parameters controlling the GPT autoencoder architecture.
    """
    vocab_size = 61
    block_size = 60
    n_embd = 512
    n_layer = 2
    n_head = 8
    dropout = 0.1
    bias = False
    n_latent_token = 2
    n_latent = 32
# end Config


# GPT Autoencodeur
class GPTAE(nn.Module):
    """
    Transformer based autoencoder with a learned latent bottleneck.

    The encoder and decoder share the same block configuration and the model
    exposes helpers for loading pretrained weights and estimating compute cost.
    """

    def __init__(
            self,
            config,
            use_flash: bool = True
    ):
        """
        Args:
            config (GPTAEConfig): Architecture description.
            use_flash (bool): Whether to enable flash-attention blocks when
                available on the current device.
        """
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        # Create transformer encodeur components
        self.encoder = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, use_flash) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias)
            )
        )

        # Projection to latent space (per token)
        self.to_latent_token = nn.Linear(config.n_embd, config.n_latent_token)

        # Projection to latent space
        self.to_latent = nn.Linear(config.n_latent_token * config.block_size, config.n_latent)

        # Projection from the latent space
        self.from_latent = nn.Linear(config.n_latent, config.n_latent_token * config.block_size)

        # Projection from latent (per token)
        self.from_latent_token = nn.Linear(config.n_latent_token, config.n_embd)

        # Create transformer decoder components
        self.decoder = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(config.block_size, config.n_embd),  # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, use_flash) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),  # final layer norm
            )
        )

        # --- Output layer ---
        self.output_head = nn.Linear(config.n_embd, config.vocab_size)
    # end def __init__

    @classmethod
    def from_pretrained(
            cls,
            repo_id: str,
            revision: str = "main",
            device: str = "cpu",
            **kwargs
    ):
        """
        Load a pretrained GPT-2 model from a HuggingFace.
        """
        # Download files
        config_path = hf_hub_download(repo_id, filename="config.yaml", subfolder="safetensors", revision=revision)
        weights_path = hf_hub_download(repo_id, filename="model.safetensors", subfolder="safetensors", revision=revision)

        # Read the configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # end with

        config = GPTConfig(**config)

        # Instantiate the model
        model = cls(
            config=config,
            **kwargs
        )

        # Load file
        state_dict = load_file(weights_path, device=device)
        model.load_state_dict(state_dict, strict=False)

        return model, config
    # end def load_pretrained

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.

        Args:
            non_embedding (bool): Whether to exclude embedding parameters from the count

        Returns:
            int: Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.wpe.weight.numel()
            n_params -= self.decoder.wpe.weight.numel()
        # end if
        return n_params
    # end def get_num_params

    def estimate_mfu(
            self,
            n_layer: int,
            n_head: int,
            n_embd: int,
            n_latent_token: int,
            n_latent: int,
            block_size: int,
            fwdbwd_per_iter,
            dt
    ):
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        This method calculates how efficiently the model is using the available
        computational resources during training.

        Args:
            module (nn.Module): Module whose parameters to configure
            n_layer (int): Number of layers
            n_head (int): Number of heads
            n_embd (int): Number of embeddings
            n_latent_token (int): Number of tokens
            n_latent (int): Number of latent variables
            block_size (int): Block size
            fwdbwd_per_iter (int): Number of forward-backward passes per iteration
            dt (float): Time per iteration in seconds

        Returns:
            float: Model FLOPs utilization as a fraction of theoretical peak
        """
        # First estimate the number of flops we do per iteration.
        # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = n_layer, n_head, n_embd // n_head, block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_token *= 2
        flops_per_token += n_embd * n_latent_token * 2
        flops_per_token += n_latent * n_latent_token * 2
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu  # end def estimate_mfu
    # end def estimate_mfu

    def forward_transformer(
            self,
            module: nn.Module,
            idx: torch.LongTensor
    ):
        """
        Run either the encoder or decoder Transformer stack on token indices.

        Args:
            module (nn.Module): Module dict that exposes ``wte``, ``wpe``, ``drop``,
                ``h`` (a list of blocks) and ``ln_f``.
            idx (torch.LongTensor): Token indices shaped ``(batch, seq_len)``.

        Returns:
            torch.Tensor: Final hidden states of shape ``(batch, seq_len, n_embd)``.
        """
        device = idx.device

        # Batch size, sequence length
        b, t = idx.size()

        # Check max sequence length
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Pos is (t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward the GPT model itself
        # tok_emb is (b, t, 512)
        tok_emb = module.wte(idx)  # token embeddings of shape (b, t, n_embd)

        # pos_emb is (b, t, 512)
        pos_emb = module.wpe(pos)  # position embeddings of shape (t, n_embd)

        # x is (b, t, 512)
        x = module.drop(tok_emb + pos_emb)

        # Process through transformer blocks
        for (block_i, block) in enumerate(module.h):
            x = block(x=x)
            # output is (b, t, 512)
        # end for

        # Layer norm
        # output is (b, t, 512)
        x = module.ln_f(x)

        return x
    # end def forward_encoder

    def forward_decoder(
            self,
            x: torch.Tensor,
    ):
        """
        Decode latent embeddings with a Transformer decoder stack.

        Args:
            x (torch.Tensor): Latent token embeddings shaped ``(batch, seq_len, n_embd)``.

        Returns:
            torch.Tensor: Decoder hidden states shaped ``(batch, seq_len, n_embd)``.
        """
        device = x.device

        # Batch size, sequence length
        b, t, e = x.size()

        # Check max sequence length
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Pos is (t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # pos_emb is (b, t, 512)
        pos_emb = self.decoder.wpe(pos)  # position embeddings of shape (t, n_embd)

        # x is (b, t, 512)
        x = self.decoder.drop(x + pos_emb)

        # Process through transformer blocks
        for (block_i, block) in enumerate(self.decoder.h):
            x = block(x=x)
            # output is (b, t, 512)
        # end for

        # Layer norm
        # output is (b, t, 512)
        x = self.decoder.ln_f(x)

        return x
    # end def forward_decoder

    def forward(self, idx: torch.LongTensor):
        """
        Args:
            idx (torch.LongTensor): Input sequence indices.

        Returns:
            logits: FloatTensor (batch, seq_len, vocab_size)
        """
        # Encoder
        x = self.forward_transformer(
            module=self.encoder,
            idx=idx
        )

        # Compression
        z = self.to_latent_token(x)  # (B, L, n_latent)

        # Size
        B, L, n_latent_token = z.shape

        # Compression 2
        z = z.view(B, L * n_latent_token)
        z = self.to_latent(z)

        # Expansion 1
        x = self.from_latent(z)
        x = x.view(B, L, n_latent_token)

        # Expansion 2
        x = self.from_latent_token(x)  # (B, L, 512)

        # Decoder
        x = self.forward_decoder(x)  # (B, L, 512)

        # Output logits
        logits = self.output_head(x)  # (B, L, vocab_size)

        return logits
    # end def forward

    def encode(self, idx: torch.LongTensor):
        """
        Encode the input sequence.

        Args:
            idx (torch.LongTensor): Input sequence indices.

        Returns:
            Encoding vector (B, L, n_latent_token)
        """
        assert idx.ndim == 2, f"idx must have shape [batch_size, seq_len], got {idx.shape}"

        # To device
        idx = idx.to(next(self.parameters()).device)

        # Encoder
        x = self.forward_transformer(
            module=self.encoder,
            idx=idx
        )

        # Compression
        z = self.to_latent_token(x)  # (B, L, n_latent)

        # Size
        B, L, n_latent_token = z.shape

        # Compression 2
        z = z.view(B, L * n_latent_token)
        return self.to_latent(z)
    # end encode

    def decode(self, emb: torch.Tensor):
        """
        Decode an embedding

        Args:
            emb: torch.Tensor,

        Returns:
            logits: FloatTensor (batch, seq_len, vocab_size)
        """
        assert emb.ndim == 2, f"emb must have shape [batch_size, {self.config.n_latent}], got {emb.shape}"

        # To device
        emb = emb.to(next(self.parameters()).device)

        # Size
        B, _ = emb.shape

        # From latent to latent token
        x = self.from_latent(emb)

        # View
        x = x.view(B, self.config.block_size, self.config.n_latent_token)

        # From latent token
        x = self.from_latent_token(x)

        # Decode
        x = self.forward_decoder(x)

        # Output logits
        logits = self.output_head(x)  # (B, L, vocab_size)

        return logits
    # end decode

# end class GPTAE

