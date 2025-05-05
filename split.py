#!/usr/bin/env python3
import os
import glob
import pandas as pd
import argparse
import math

def split_csv_to_txts_sliding(csv_path, out_dir,
                              sample_rate=6250,
                              window_sec=1.0,
                              step_sec=0.01,
                              skiprows=4):
    """
    Read `csv_path`, skip first `skiprows` lines, take only even columns,
    then slide a `window_sec`-long window every `step_sec`, writing each
    segment of length window_sec*sample_rate rows to a .txt.
    """
    fname = os.path.basename(csv_path)
    name, _ = os.path.splitext(fname)

    # read the data, no header
    df = pd.read_csv(csv_path, skiprows=skiprows, header=None)

    # keep only 2nd, 4th, 6th, ... columns (1-based even)
    df = df.iloc[:, 1::2]

    total_rows = len(df)
    segment_length = int(round(sample_rate * window_sec))
    step_length    = int(round(sample_rate * step_sec))

    if total_rows < segment_length:
        print(f"[!] {fname}: only {total_rows} rows, less than one window ({segment_length}); skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for start in range(0, total_rows - segment_length + 1, step_length):
        seg = df.iloc[start:start + segment_length]
        count += 1
        out_name = f"{name}_{count:04d}.txt"
        out_path = os.path.join(out_dir, out_name)
        # space-separated, no header/index
        seg.to_csv(out_path, sep=' ', header=False, index=False)

    print(f"[+] {fname}: wrote {count} overlapping windows of {segment_length} rows each to {out_dir}")

def main():
    p = argparse.ArgumentParser(
        description="Slide 1s (6.25k rows) window every 0.01s (≈62 rows) over x_y_t.csv files")
    p.add_argument("in_dir",  help="Directory containing x_y_t.csv files")
    p.add_argument("out_dir", help="Directory to write the .txt windows")
    args = p.parse_args()

    pattern = os.path.join(args.in_dir, "*_*_*.csv")
    files = glob.glob(pattern)
    if not files:
        print("No files matching x_y_t.csv found in", args.in_dir)
        return

    for f in files:
        split_csv_to_txts_sliding(f, args.out_dir)

if __name__ == "__main__":
    main()
