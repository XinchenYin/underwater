# train.py
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from underwater_dataset import UnderwaterDataset
from model import AMSS_FFN

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2))

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        x_tf = batch['image'].to(device)      # [B,3,H,W]
        x_t  = batch['signal'].to(device)     # [B,4,6250]
        y    = batch['position'].to(device)   # [B,2]

        optimizer.zero_grad()
        preds = model(x_tf, x_t)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_tf.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    for batch in tqdm(loader, desc="Val", leave=False):
        x_tf = batch['image'].to(device)
        x_t  = batch['signal'].to(device)
        y    = batch['position'].to(device)

        preds = model(x_tf, x_t)
        loss = criterion(preds, y)
        val_loss += loss.item() * x_tf.size(0)
        total_mae  += mae(preds, y).item() * x_tf.size(0)
        total_rmse += rmse(preds, y).item() * x_tf.size(0)

    n = len(loader.dataset)
    return val_loss/n, total_mae/n, total_rmse/n

def main(args):
    # Initialize W&B
    wandb.init(project="AMSS-FFN-underwater", config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler_gamma": args.gamma
    })
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets & loaders
    train_ds = UnderwaterDataset(
        txt_dir=args.txt_dir, img_dir=args.img_dir,
        txt_transform=lambda arr: torch.from_numpy((arr - arr.mean())/arr.std()).float(), img_transform=None
    )
    # split train/val
    n = len(train_ds)
    n_train = int(0.7 * n)
    n_val   = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [n_train, n_val]
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    # Model, loss, optimizer, scheduler
    model = AMSS_FFN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.scheduler_gamma
    )

    wandb.watch(model, log="all")

    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_mae, val_rmse = validate(
            model, val_loader, criterion, device
        )

        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_MAE": val_mae,
            "val_RMSE": val_rmse,
            "lr": scheduler.get_last_lr()[0]
        })

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Val RMSE: {val_rmse:.4f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    print("Training complete. Best val loss: {:.4f}".format(best_val))
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_dir", type=str, required=True,
                        help="Folder with .txt signals")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Folder with .png time–frequency images")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max number of epochs")               # paper: 100 epochs :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")                         # paper: 8 :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")               # paper: 0.001 :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="LR decay per epoch")                 # paper: decay rate 0.95 :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience")            # paper: patience=8 :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth",
                        help="Where to save the best model")
    args = parser.parse_args()

    main(args)
