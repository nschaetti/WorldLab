
import argparse
import json

import torch
from safetensors.torch import save_model
from worldlab.nn.board import GPTConfig, GPTAEConfig
from worldlab.nn import GameGPT, GameAutoEncoder


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--safetensors", type=str, required=True, help="Output safetensors file")
    parser.add_argument("--model-type", type=str, required=True, help="Model type")
    parser.add_argument("--config", type=str, required=True, help="Output JSON config file")
    parser.add_argument("--n-layer", type=int, default=8, help="How many layers the model has")
    parser.add_argument('--n-head', type=int, default=8, help="How many heads the model has")
    parser.add_argument('--n-embd', type=int, default=512, help="How many embedding dimensions the model has")
    parser.add_argument('--block-size', type=int, default=61, help="How many tokens in the input")
    parser.add_argument('--bias', action='store_true', default=False, help="Whether to add a bias to the output")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout probability")
    parser.add_argument('--n-latent-token', type=int, default=None, help="Size of the latent space of each token")
    parser.add_argument('--n-latent', type=int, default=None, help="Size of the latent space.")
    args = parser.parse_args()

    # Set up model arguments from configuration
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        dropout=args.dropout,
    )

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']

    # Force these config attributes to be equal otherwise we can't resume training
    # The rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'n_latent', 'n_latent_token']:
        model_args[k] = checkpoint_model_args[k]
    # end for

    # Create the model
    if args.model_type.lower() == 'gpt':
        gptconf = GPTConfig(**model_args)
        model = GameGPT(gptconf)
    elif args.model_type.lower() == 'autoencoder':
        assert args.n_latent_token is not None
        assert args.n_latent is not None
        model_args['n_latent_token'] = args.n_latent_token
        model_args['n_latent'] = args.n_latent
        gptconf = GPTConfig(**model_args)
        model = GameAutoEncoder(gptconf)
    else:
        raise ValueError(f"Unknown model type {args.model_type}")
    # end if

    # Load the state dict
    state_dict = checkpoint['model']

    # Fix the keys of the state dictionary if needed
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        # end if
    # end for

    # Load weights
    model.load_state_dict(state_dict)

    # Save tensor file
    save_model(model, args.safetensors)

    # Save json config
    with open(args.config, 'w') as f:
        json.dump(model_args, f)
    # end with
# end if

