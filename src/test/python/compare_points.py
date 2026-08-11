#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""Compare two csvs of transformed points, printing them side by side.

Exits nonzero if any coordinate differs by more than the tolerance. Used by
compare_java_scipy.sh to check jitk's Java thin plate spline against scipy.

Usage:
  compare_points.py points.csv a.csv b.csv [--labels A B] [--tolerance T]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tps_rbf import read_numeric_csv


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Compare two csvs of transformed points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("points", help="csv of the input points")
    parser.add_argument("a", help="csv of transformed points")
    parser.add_argument("b", help="csv of transformed points to compare against")
    parser.add_argument("--labels", nargs=2, default=["a", "b"],
        help="column headings for the two transformed csvs")
    parser.add_argument("--tolerance", type=float, default=1e-9,
        help="max absolute difference tolerated before failing")
    parsed = parser.parse_args(args)

    try:
        points = read_numeric_csv(parsed.points)
        a = read_numeric_csv(parsed.a)
        b = read_numeric_csv(parsed.b)
    except (OSError, ValueError) as e:
        parser.error(str(e))

    if not (len(points) == len(a) == len(b)):
        parser.error(f"row counts differ: {len(points)} points, "
            f"{len(a)} in {parsed.a}, {len(b)} in {parsed.b}")
    if a.shape != b.shape:
        parser.error(f"shapes differ: {a.shape} vs {b.shape}")

    width = 13 * points.shape[1] - 1
    row_fmt = "{:>%d}   {:>%d}   {:>%d}   {:>10}" % (width, width, width)
    print(row_fmt.format("point", parsed.labels[0], parsed.labels[1], "max|diff|"))

    def cols(row):
        return " ".join(f"{v:>12.6f}" for v in row)

    diff = np.abs(a - b)
    for p, av, bv, d in zip(points, a, b, diff):
        print(f"{cols(p)}   {cols(av)}   {cols(bv)}   {d.max():10.3e}")

    print()
    print(f"max absolute difference: {diff.max():.3e} "
        f"(tolerance {parsed.tolerance:.3e})")
    if diff.max() > parsed.tolerance:
        print(f"FAIL: {parsed.labels[0]} and {parsed.labels[1]} disagree")
        return 1
    print(f"OK: {parsed.labels[0]} matches {parsed.labels[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
