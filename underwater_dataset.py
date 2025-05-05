import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class UnderwaterDataset(Dataset):
    def __init__(self,
                 txt_dir: str,
                 img_dir: str,
                 txt_transform=None,
                 img_transform=None):
        """
        txt_dir: folder containing .txt files
        img_dir: folder containing .png files
        txt_transform: optional fn(signal: np.ndarray) -> Tensor
        img_transform: optional torchvision transform for PIL Image
        """
        self.txt_dir = txt_dir
        self.img_dir = img_dir
        self.txt_transform = txt_transform
        self.img_transform = img_transform

        # map basenames (without ext) to paths
        txt_paths = glob.glob(os.path.join(txt_dir, '*.txt'))
        img_paths = glob.glob(os.path.join(img_dir, '*.png'))
        txt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in txt_paths}
        img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}

        # only keep files that appear in both dirs
        common = sorted(set(txt_map) & set(img_map))
        if not common:
            raise RuntimeError(f"No matching files in {txt_dir} and {img_dir}")

        # store list of (txt_path, img_path, basename)
        self.samples = [(txt_map[b], img_map[b], b) for b in common]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        txt_path, img_path, basename = self.samples[idx]

        # --- load text data ---
        signal = np.loadtxt(txt_path)              # shape: (T, N_sensors)
        if self.txt_transform:
            signal = self.txt_transform(signal)
        else:
            # default: convert to FloatTensor
            signal = torch.from_numpy(signal).float()

        # --- load image data ---
        img = Image.open(img_path).convert('RGB')
        if self.img_transform:
            img = self.img_transform(img)
        else:
            arr = np.array(img)                     # H×W×3, uint8
            img = torch.from_numpy(arr).permute(2,0,1).float().div(255.)

        # --- parse x,y from filename ---
        # assume basename = "x_y_..." or "x_y"
        parts = basename.split('_')
        x = float(parts[0])
        y = float(parts[1])
        position = torch.tensor([x, y], dtype=torch.float)

        return {
            'signal':    signal,    # FloatTensor [T, N_sensors]
            'image':     img,       # FloatTensor [3, H, W]
            'position':  position,  # FloatTensor [2]
        }

# --- Example usage ---
if __name__ == "__main__":
    ds = UnderwaterDataset(
        txt_dir='data/norm_time',
        img_dir='data/freq_domain',
        # you can pass e.g. a lambda to normalize your signal:
        txt_transform=lambda arr: torch.from_numpy((arr - arr.mean())/arr.std()).float(),
        # or standard torchvision transforms for your image:
        img_transform=None
    )

    print("Dataset size:", len(ds))
    sample = ds[0]
    print("signal shape:", sample['signal'].shape)
    print("image shape:",  sample['image'].shape)
    print("position:",     sample['position'])
