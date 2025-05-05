#!/usr/bin/env python3
import os
import glob
import numpy as np
import argparse

def normalize_time_segment(txt_path, out_dir):
    """
    Load a 1s segment (6250×4) from txt_path, apply:
      1) bottom-noise removal: x' = x - mean(x)    (Eq. 13)
      2) global normalization: y = 255*(x' - xmin)/(xmax - xmin)  (Eq. 14)
    Then save to out_dir with suffix '_norm.txt'.
    """
    fname = os.path.basename(txt_path)
    name, _ = os.path.splitext(fname)

    # load space-separated floats
    data = np.loadtxt(txt_path)

    # 1) subtract per-sensor (column) mean
    data = data - np.mean(data, axis=0)                   # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

    # 2) compute global min/max over all sensors & samples
    xmin, xmax = data.min(), data.max()
    if np.isclose(xmax, xmin):
        raise ValueError(f"{fname}: constant data, cannot normalize")

    # map into [0,255]
    normed = 255.0 * (data - xmin) / (xmax - xmin)        # :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.txt")

    # save with 6 decimal places
    np.savetxt(out_path, normed, fmt="%.6f", delimiter=" ")
    print(f"[+] {fname} → {os.path.basename(out_path)}")

def main():
    parser = argparse.ArgumentParser(
        description="Normalize 1s time-domain TXT segments into [0,255]")
    parser.add_argument("in_dir", help="Directory of raw .txt segments")
    parser.add_argument("out_dir", help="Directory for normalized outputs")
    args = parser.parse_args()

    pattern = os.path.join(args.in_dir, "*_*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No .txt files found in {args.in_dir}")
        return

    for txt in files:
        normalize_time_segment(txt, args.out_dir)

if __name__ == "__main__":
    main()
