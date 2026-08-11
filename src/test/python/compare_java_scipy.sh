#!/usr/bin/env bash
# Fit a thin plate spline from landmark pairs with both this repo's Java
# implementation (jitk.spline.TpsCsvApply) and scipy's RBFInterpolator
# (tps_rbf.py), transform the same points with each, and compare.
#
# usage: compare_java_scipy.sh [landmarks.csv [points.csv [out_dir]]]
#
# Defaults to the sample data in src/test/resources. The transformed points are
# written to out_dir as java.csv and scipy.csv and kept there; without out_dir
# they go to a temporary directory that is deleted on exit.
#
# The python side runs under uv, which resolves numpy and scipy from the
# scripts' inline (PEP 723) dependencies. Environment:
#   UV     uv executable to use (default uv, from PATH)
#   TOL    max allowed absolute difference (default 1e-9)
#   NDIMS  number of dimensions (default: column count of the points csv)

set -euo pipefail

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"
repo_root="$( cd -- "$script_dir/../../.." && pwd )"

landmarks="${1:-$repo_root/src/test/resources/landmarks-2d.csv}"
points="${2:-$repo_root/src/test/resources/points-2d.csv}"
out_dir="${3:-}"
uv_bin="${UV:-uv}"
tol="${TOL:-1e-9}"

for f in "$landmarks" "$points"; do
	if [ ! -f "$f" ]; then
		echo "no such file: $f" >&2
		exit 1
	fi
done

if ! command -v "$uv_bin" > /dev/null; then
	echo "uv not found on PATH; install it (https://docs.astral.sh/uv/) or set UV" >&2
	exit 1
fi

# Compile the Java sources and cache the runtime classpath, refreshing it
# whenever the pom changes.
classpath_file="$repo_root/target/test-classpath.txt"
echo "building $repo_root ..." >&2
mvn -q -f "$repo_root/pom.xml" test-compile
if [ ! -s "$classpath_file" ] || [ "$repo_root/pom.xml" -nt "$classpath_file" ]; then
	mvn -q -f "$repo_root/pom.xml" dependency:build-classpath \
		"-Dmdep.outputFile=$classpath_file"
fi
classpath="$repo_root/target/test-classes:$repo_root/target/classes:$( cat "$classpath_file" )"

if [ -n "$out_dir" ]; then
	mkdir -p "$out_dir"
else
	# Nowhere was asked for, so clean up after ourselves.
	out_dir="$( mktemp -d )"
	trap 'rm -rf "$out_dir"' EXIT
fi
java_out="$out_dir/java.csv"
scipy_out="$out_dir/scipy.csv"

echo "landmarks: $landmarks" >&2
echo "points:    $points" >&2
echo "output:    $out_dir" >&2

java -cp "$classpath" jitk.spline.TpsCsvApply \
	"$landmarks" "$points" ${NDIMS:+"$NDIMS"} > "$java_out"

"$uv_bin" run --script "$script_dir/tps_rbf.py" "$landmarks" "$points" \
	${NDIMS:+--ndims "$NDIMS"} --precision 17 -o "$scipy_out"

"$uv_bin" run --script "$script_dir/compare_points.py" \
	"$points" "$java_out" "$scipy_out" --labels java scipy --tolerance "$tol"
