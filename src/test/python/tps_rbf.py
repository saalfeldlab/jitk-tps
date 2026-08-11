#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scipy>=1.7"]
# ///
"""Transform points with a thin plate spline fit by scipy's RBFInterpolator.

This is a reference implementation to check jitk's
jitk.spline.ThinPlateR2LogRSplineKernelTransform against. The two solve the same
system:

  * kernel 'thin_plate_spline' is r**2 * log(r), the same kernel jitk uses (in
    every dimension, so 3D comparisons are valid too).
  * degree=1 adds the affine polynomial term, matching computeAffine=true.
  * smoothing corresponds to jitk's `stiffness` field (hardcoded to 0.0, i.e.
    exact interpolation), which is added to the diagonal of the kernel matrix.

jitk's kernel is the matrix r**2*log(r) * I, which is block diagonal, so each
output dimension is solved independently -- exactly what scipy does per
component.

Gotcha when comparing: jitk stores landmarks dimension-major, as
double[ndims][nLandmarks] (see ThinPlateR2LogRSplineKernelTransform:120), while
the CSVs here and scipy are both point-major (npoints x ndims). A missing
transpose is the most likely cause of a bogus mismatch.

Dependencies are declared inline (PEP 723), so run it with uv:

  uv run --script tps_rbf.py landmarks.csv points.csv

or just ./tps_rbf.py landmarks.csv points.csv, since the shebang does the same.
Here landmarks.csv holds the moving coordinates in the first ndims columns and
the corresponding target coordinates in the next ndims columns, and points.csv
holds moving points to transform, one per row. A non-numeric first row (a
header) is skipped in both files. Transformed coordinates are written to stdout.
"""

import argparse
import sys

import numpy as np


def read_numeric_csv(path, delimiter=","):
    """Read a numeric csv into an (nrows, ncols) float64 array.

    Blank lines and '#' comments are dropped, as is a leading header row (any
    first row that does not parse as numbers).
    """
    with open(path) as f:
        rows = []  # (line number, columns)
        for lineno, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].strip()
            if line:
                rows.append((lineno, line.split(delimiter)))

    if not rows:
        raise ValueError(f"{path}: no data rows")

    try:
        [float(v) for v in rows[0][1]]
    except ValueError:
        rows = rows[1:]  # header
        if not rows:
            raise ValueError(f"{path}: no data rows after the header")

    ncols = len(rows[0][1])
    for lineno, row in rows:
        if len(row) != ncols:
            raise ValueError(
                f"{path} line {lineno}: has {len(row)} columns, expected {ncols}")
        for v in row:
            try:
                float(v)
            except ValueError:
                raise ValueError(f"{path} line {lineno}: {v!r} is not a number")

    return np.array([row for _, row in rows], dtype=np.float64)


def fit(landmarks, ndims, smoothing=0.0, kernel="thin_plate_spline", degree=None):
    """Fit a TPS mapping the moving landmarks to the target landmarks."""
    # Imported here so that read_numeric_csv can be reused by scripts that only
    # declare numpy as a dependency (see compare_points.py).
    from scipy.interpolate import RBFInterpolator

    return RBFInterpolator(
        landmarks[:, :ndims],
        landmarks[:, ndims:2 * ndims],
        neighbors=None,  # use all landmarks, as jitk does
        smoothing=smoothing,
        kernel=kernel,
        degree=degree)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Transform points with a thin plate spline fit from landmark pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("landmarks",
        help="csv of landmarks: ndims moving columns then ndims target columns")
    parser.add_argument("points",
        help="csv of moving points to transform, ndims columns")
    parser.add_argument("-o", "--output", default=None,
        help="write transformed points here instead of stdout")
    parser.add_argument("--ndims", type=int, default=None,
        help="number of dimensions (default: half the landmark column count)")
    parser.add_argument("--smoothing", type=float, default=0.0,
        help="regularization; 0 interpolates exactly (jitk's stiffness)")
    parser.add_argument("--kernel", default="thin_plate_spline",
        help="scipy RBF kernel; r**2*log(r) is jitk's")
    parser.add_argument("--degree", type=int, default=None,
        help="degree of the added polynomial (default: 1 for thin_plate_spline)")
    parser.add_argument("--precision", type=int, default=9,
        help="significant digits printed")
    parser.add_argument("--delimiter", default=",",
        help="column delimiter, for both input and output")
    parsed = parser.parse_args(args)

    try:
        landmarks = read_numeric_csv(parsed.landmarks, parsed.delimiter)
        points = read_numeric_csv(parsed.points, parsed.delimiter)
    except (OSError, ValueError) as e:
        parser.error(str(e))

    ndims = parsed.ndims
    if ndims is None:
        if landmarks.shape[1] % 2 != 0:
            parser.error(
                f"{parsed.landmarks} has an odd number of columns "
                f"({landmarks.shape[1]}); pass --ndims")
        ndims = landmarks.shape[1] // 2
    if ndims < 1:
        parser.error(f"--ndims must be positive, got {ndims}")
    if landmarks.shape[1] < 2 * ndims:
        parser.error(
            f"{parsed.landmarks} has {landmarks.shape[1]} columns, "
            f"need at least {2 * ndims} for {ndims}d landmark pairs")
    if points.shape[1] != ndims:
        parser.error(
            f"{parsed.points} has {points.shape[1]} columns, "
            f"expected {ndims} for {ndims}d points")

    interp = fit(landmarks, ndims,
        smoothing=parsed.smoothing, kernel=parsed.kernel, degree=parsed.degree)
    transformed = interp(points)

    fmt = "{:." + str(parsed.precision) + "g}"
    lines = [parsed.delimiter.join(fmt.format(v) for v in pt) for pt in transformed]
    text = "\n".join(lines) + "\n"
    if parsed.output is None:
        sys.stdout.write(text)
    else:
        with open(parsed.output, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
