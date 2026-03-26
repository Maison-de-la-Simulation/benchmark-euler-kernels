import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Display a file.")

    parser.add_argument("filename", type=str, help="Path to the file to display")

    args = parser.parse_args()

    # Example usage of the filename
    print(f"Processing file: {args.filename}")

    array = np.load(args.filename)
    plt.imshow(array[int(array.shape[0] / 2)], origin="lower")
    plt.colorbar()
    plt.show()


import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def main_loop():
    parser = argparse.ArgumentParser(description="Display a sequence of .npy files.")
    parser.add_argument("path", type=str, help="Directory or glob pattern (e.g. './*.npy')")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between frames (seconds)")
    args = parser.parse_args()

    # Resolve files
    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, "*.npy"))
    else:
        files = glob.glob(args.path)

    if not files:
        raise RuntimeError("No .npy files found")

    # Sort numerically based on timestep in filename
    files.sort()

    print(f"Found {len(files)} files")

    plt.ion()  # interactive mode
    fig, ax = plt.subplots()

    im = None
    i = 0
    while True:
        i = i % len(files)
        f = files[i]


        print(f"Loading {f}")

        try:
            arr = np.load(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
        
        slice_ = arr[arr.shape[0] // 2]
        print(f"{f}: shape={arr.shape}, min={arr.min()}, max={arr.max()}, any NaN={np.isnan(arr).any()}")

        slice_ = arr[arr.shape[0] // 2]

        print(f"Slice {arr.shape[0]//2}: min={slice_.min()}, max={slice_.max()}, any NaN={np.isnan(slice_).any()}")

        if im is None:
            im = ax.imshow(slice_, origin="lower")
            plt.colorbar(im, ax=ax)
        else:
            im.set_data(slice_)
            im.set_clim(vmin=slice_.min(), vmax=slice_.max())

        ax.set_title(os.path.basename(f))
        plt.pause(args.delay)
        i += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main_loop()
    # main()
