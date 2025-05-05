import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from stockwell import st

def s_transform_grid(data, output_path, fmin=0, fmax=100, df=1.0, px_per_side=256):
    """
    Given a 2D array `data` of shape (n_samples, 4),  
    compute and save a 2×2 grid of S-transform spectrograms.

    Parameters
    ----------
    data : np.ndarray, shape (N, 4)
    output_path : str
        Full path to save the 256×256 PNG.
    fmin, fmax : float
        Frequency bounds (Hz).
    df : float
        Frequency step (Hz).
    px_per_side : int
        Pixel dimension (both width and height) of the output image.
    """
    if data.shape[1] != 4:
        raise ValueError(f"Expected data with 4 columns (sensors), got {data.shape[1]}.")

    # Convert freq bounds to sample indices
    fmin_idx = int(fmin / df)
    fmax_idx = int(fmax / df)

    # Figure size & DPI so that figsize*DPI = px_per_side
    # e.g. 256px / 64dpi = 4in
    dpi = 64
    figsize = (px_per_side / dpi, px_per_side / dpi)

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    t = np.arange(data.shape[0])
    extent = (t[0], t[-1], fmin, fmax)

    for idx, ax in enumerate(axes.flat):
        h = data[:, idx].astype(float)
        h -= h.mean()
        S = st.st(h, fmin_idx, fmax_idx)
        ax.imshow(np.abs(S),
                  origin='lower',
                  extent=extent,
                  aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])

    # zero whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def process_directory(input_dir, output_dir):
    """
    Iterate over all .txt files in input_dir, and for each:
      - load the 4-column time series
      - compute & save a 256×256px 2×2 grid S-transform image

    Parameters
    ----------
    input_dir : str
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(input_dir, '*.txt')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No .txt files found in {input_dir}")
        return

    for fpath in files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        outname = f"{basename}.png"
        outpath = os.path.join(output_dir, outname)
        if os.path.exists(outpath):
            print(f"✖ Skipped {outname} (already exists)")
            continue

        # Load and process
        data = np.loadtxt(fpath)
        try:
            s_transform_grid(data, outpath)
            print(f"✔ Saved {outname}")
        except Exception as e:
            print(f"✖ Failed {basename}: {e}")


if __name__ == "__main__":
    # Customize these two paths as needed:
    input_folder = 'data/split_time'
    output_folder = 'data/freq_domain'
    process_directory(input_folder, output_folder)
