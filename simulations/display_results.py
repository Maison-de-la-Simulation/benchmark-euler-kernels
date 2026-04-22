"""
Simple CLI tool to load and visualize a NumPy array file.

This script accepts the path to a `.npy` file as a command-line argument,
loads the array using NumPy, and displays a 2D slice using Matplotlib.
Specifically, it visualizes the middle slice along the first axis of
the array and includes a colorbar for reference.

Example:
    python script.py data.npy
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Parse command-line arguments and display a slice of a NumPy array.

    This function:
    1. Parses a filename from the command line.
    2. Loads the NumPy array stored in the file.
    3. Extracts the middle slice along the first dimension.
    4. Displays the slice as an image using Matplotlib.
    """
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
