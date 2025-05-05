import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict

from underwater_dataset import UnderwaterDataset
from model import AMSS_FFN

def mae(pred, target):
    return np.mean(np.abs(pred - target))

def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

def build_subsets_by_position(dataset, samples_per_pos, seed=42):
    """
    Group dataset by position and sample up to `samples_per_pos` for each position.
    Returns a dict mapping position string -> Subset.
    """
    # Map position -> list of indices
    pos2idx = defaultdict(list)
    for idx, (_, _, basename) in enumerate(dataset.samples):
        pos = "_".join(basename.split("_")[:2])
        pos2idx[pos].append(idx)

    subsets = {}
    random.seed(seed)
    for pos, idxs in pos2idx.items():
        if samples_per_pos is not None:
            if len(idxs) < samples_per_pos:
                raise ValueError(f"Position {pos} only has {len(idxs)} samples, need {samples_per_pos}.")
            selected = random.sample(idxs, samples_per_pos)
        else:
            selected = idxs
        subsets[pos] = Subset(dataset, selected)
    return subsets

@torch.no_grad()
def evaluate(model, loader, device):
    all_preds = []
    all_targets = []
    for batch in loader:
        x_tf = batch['image'].to(device)
        x_t  = batch['signal'].to(device)
        y    = batch['position'].to(device)
        outputs = model(x_tf, x_t)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return mae(preds, targets), rmse(preds, targets), preds, targets


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full test dataset
    full_ds = UnderwaterDataset(
        txt_dir=args.txt_dir,
        img_dir=args.img_dir,
        txt_transform=None,
        img_transform=None
    )

    # Build per-position subsets
    subsets = build_subsets_by_position(full_ds, args.samples_per_pos)
    print(f"Built subsets for {len(subsets)} positions; each with {args.samples_per_pos or 'all'} samples.")

    # Load model
    model = AMSS_FFN().to(device)
    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Evaluate each position separately
    summary = []
    for pos, subset in subsets.items():
        loader = DataLoader(subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
        pos_mae, pos_rmse, preds, targets = evaluate(model, loader, device)
        print(f"Position {pos} -> MAE: {pos_mae:.4f}, RMSE: {pos_rmse:.4f}")
        summary.append((pos, pos_mae, pos_rmse))

        # save per-position results if requested
        if args.save_results:
            import pandas as pd
            df = pd.DataFrame({
                'x_true':   targets[:,0],
                'y_true':   targets[:,1],
                'x_pred':   preds[:,0],
                'y_pred':   preds[:,1]
            })
            out_csv = f"{pos}_results.csv"
            df.to_csv(out_csv, index=False)
            print(f"Saved results for {pos} to {out_csv}")

    # Optionally, aggregate overall metrics
    all_mae = np.mean([m for _, m, _ in summary])
    all_rmse = np.mean([r for _, _, r in summary])
    print(f"\nAverage over positions -> MAE: {all_mae:.4f}, RMSE: {all_rmse:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Per-position testing of AMSS-FFN"
    )
    parser.add_argument('--txt_dir', type=str, required=True,
                        help='Directory with .txt signal files')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory with .png image files')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the trained model .pth')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--samples_per_pos', type=int, default=500,
                        help='Samples per position (None for all)')
    parser.add_argument('--save_results', action='store_true',
                        help='Whether to save per-position CSV results')
    args = parser.parse_args()

    # Convert 'None' string to None
    if args.samples_per_pos == 0:
        args.samples_per_pos = None

    main(args)
