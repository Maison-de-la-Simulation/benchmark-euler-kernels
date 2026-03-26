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


if __name__ == "__main__":
    main()
