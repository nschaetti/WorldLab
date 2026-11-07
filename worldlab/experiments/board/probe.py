"""
Copyright (C) 2025 Nils Schaetti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import wandb
import argparse
import torch
import torch.nn as nn
import yaml

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import MeanMetric
from transformers import AutoTokenizer
from rich.console import Console
from rich.traceback import install

from worldlab.nn import BoardLinearProbe, BoardMLPProbe, GameGPT
from worldlab.training import optimizer_step
from worldlab.board.models.utils import save_board_checkpoint as save_checkpoint
from worldlab.board.datasets import BoardDataset, collate_fn_board
from worldlab.board.games.othello import game_to_board
from worldlab.board.utils import info, warning, error


console = Console()
install(width=None)


def infinite_loader(
        dataloader: torch.utils.data.DataLoader
):
    """
    Infinite loader iterator.
    """
    while True:
        for batch in dataloader:
            X, Y = batch
            yield X, Y
        # end for
    # end while
# end def infinite_loader


def create_dataset(
        args,
        tokenizer: AutoTokenizer,
):
    train_dataset = BoardDataset(
        data_dir=args.data_dir,
        board_func=game_to_board,
        split="train"
    )

    val_dataset = BoardDataset(
        data_dir=args.data_dir,
        board_func=game_to_board,
        split="val"
    )

    # Create a dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: collate_fn_board(b, tokenizer)
    )

    # Create a dataloader
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate_fn_board(b, tokenizer)
    )

    return train_dataloader, val_dataloader
# end def create_dataset


def evaluate_model(
        args,
        config,
        model: nn.Module,
        probe: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
):
    val_iter = infinite_loader(dataloader)
    criterion = nn.CrossEntropyLoss()
    metric_acc = MulticlassAccuracy(num_classes=config['n_classes']).to(device)
    metric_loss = MeanMetric().to(device)
    with torch.no_grad():
        for iter_i in range(config['eval_iters']):
            batch = next(val_iter)

            # Moves and game state
            move_idx, game_x = batch
            move_idx, game_x = move_idx.to(device), game_x.to(device)

            _, _, _, residuals = model(
                idx=move_idx,
                to_return=[
                    f"residuals{config['layer_i']}"
                ]
            )

            # Stack residuals
            residuals = residuals[0]
            preds = probe(residuals)  # [B, 60, 64, 3]
            B, T, S, C = preds.shape  # 512, 60, 64, 3

            # -------------
            # 3. Flatten for loss
            # -------------
            preds = preds.view(B, T * S, C)  # [B, 30720, 3]
            targets = game_x.unsqueeze(1).expand(-1, T, -1)  # [B, 480, 64]
            targets = targets.reshape(B, -1)  # [B, 30720]

            # -------------
            # 4. Compute loss
            # -------------
            # preds: [B, 3840, 3]
            # targets: [B, 3840]
            loss = criterion(
                preds.reshape(B * T * S, 3),  # [B*30720, 3]
                targets.reshape(B * T * S).long()  # [B*30720]
            )

            # Update metrics
            metric_loss.update(loss)
            pred_labels = preds.argmax(dim=-1)  # [B, 30720]
            metric_acc.update(pred_labels.reshape(-1), targets.reshape(-1))
        # end for
    # end with

    # Show result
    epoch_loss = metric_loss.compute().item()
    epoch_acc = metric_acc.compute().item()
    return epoch_loss, epoch_acc
# end def evaluate_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    return args, config
# end get_args


def main(args, config):
    """
    Main entry point.
    """
    # Get a device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        warning("WARNING: CUDA is not available, using CPU instead.")
    # end if

    # Get a GPT model
    model, model_config = GameGPT.from_pretrained(repo_id=config['repo_id'])
    tokenizer = AutoTokenizer.from_pretrained(config['repo_id'], subfolder="tokenizer")
    model = model.to(device)
    model.eval()

    # Get dataloaders
    train_dataloader, val_dataloader = create_dataset(args, tokenizer)

    # Create model
    if config['probe_type'].lower() == "mlp":
        probe = BoardMLPProbe(
            d_model=config['residual_size'],
            board_size=config['board_size'],
            n_classes=config['n_classes']
        )
    elif config['probe_type'].lower() == "linear":
        probe = BoardLinearProbe(
            d_model=config['residual_size'],
            board_size=config['board_size'],
            n_classes=config['n_classes']
        )
    # end if
    probe.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=config['learning_rate'])

    # Iterator
    train_iter = infinite_loader(train_dataloader)

    # Mode train
    probe.train()

    # Set up wandb logging if enabled
    if config['wandb_log']:
        assert config['wandb_project'] is not None, "WandB project must be provided."
        assert config['wandb_run_name'] is not None, "WandB run name must be provided."
        wandb.init(
            project=config['wandb_project'],
            name=config['wandb_run_name'],
            config=config
        )
    # end if

    # Metrics on GPUs
    metric_acc = MulticlassAccuracy(num_classes=config['n_classes']).to(device)
    metric_loss = MeanMetric().to(device)

    for iter_i in range(config['n_iter']):
        batch = next(train_iter)

        # Moves and game state
        move_idx, game_x = batch

        # To cuda
        move_idx, game_x = move_idx.to(device), game_x.to(device)

        # Get residuals
        with torch.no_grad():
            _, _, _, residuals = model(
                idx=move_idx,
                to_return=[
                    f"residuals{config['layer_i']}"
                ]
            )
        # end with

        # Stack residuals
        residuals = residuals[0]

        # -------------
        # 2. Forward probes
        # -------------
        # Inputs: [B, 60, 512]
        preds = probe(residuals)          # [B, 60, 64, 3]
        B, T, S, C = preds.shape        # 512, 60, 64, 3

        # -------------
        # 3. Flatten for loss
        # -------------
        preds = preds.view(B, T * S, C)           # [B, 30720, 3]
        targets = game_x.unsqueeze(1).expand(-1, T, -1)  # [B, 480, 64]
        targets = targets.reshape(B, -1)              # [B, 30720]

        # -------------
        # 4. Compute loss
        # -------------
        # preds: [B, 3840, 3]
        # targets: [B, 3840]
        loss = criterion(
            preds.reshape(B * T * S, 3),      # [B*30720, 3]
            targets.reshape(B * T * S).long()  # [B*30720]
        )

        # -------------
        # 5. Backprop
        # -------------
        loss.backward()
        optimizer_step(optimizer)

        # Update metrics
        metric_loss.update(loss)
        pred_labels = preds.argmax(dim=-1)  # [B, 30720]
        metric_acc.update(pred_labels.reshape(-1), targets.reshape(-1))

        # Show current
        if iter_i % config['log_interval'] == 0:
            acc_val = metric_acc.compute()
            loss_val = metric_loss.compute()
            info(f"Iteration {iter_i}/{config['n_iter']} - loss: {loss_val:.4f}, acc: {acc_val:.4f}")

            # Log to wandb if enabled
            if config['wandb_log']:
                log_data = {
                    "iter": iter_i,
                    "train/loss": loss_val,
                    "train/acc": acc_val,
                }
                wandb.log(log_data)
            # end if

            metric_acc.reset()
            metric_loss.reset()
        # end if

        # Eval
        if iter_i % config['eval_interval'] == 0:
            val_loss, val_acc = evaluate_model(
                args=args,
                config=config,
                model=model,
                probe=probe,
                dataloader=val_dataloader,
                device=device
            )

            info(f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

            if config['wandb_log']:
                wandb.log({
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                })
            # end if

            # Save the checkpoint
            save_checkpoint(
                out_dir=args.out_dir,
                model=probe,
                optimizer=optimizer,
                iter_num=iter_i,
                best_val_loss=val_loss,
                config=config,
                model_args={
                    "probe_type": config['probe_type'],
                    "residual_size": config['residual_size'],
                    "board_size": config['board_size'],
                    "n_classes": config['n_classes'],
                }
            )
        # end if
    # end for

    info(f"Epoch done - loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

    if config['wandb_log']:
        wandb.summary["loss"] = loss_val
        wandb.summary["acc"] = acc_val
    # end if
    metric_acc.reset()
    metric_loss.reset()
# end def main


def cli():
    args, config = get_args()
    main(args, config)


if __name__ == "__main__":
    cli()
