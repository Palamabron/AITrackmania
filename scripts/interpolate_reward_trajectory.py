"""
Interpolate a reward trajectory pkl to add more points (e.g. 10x) along the track.
Finer spacing gives a more granular progress signal (e.g. on difficult turns)
without changing total reward scale: progress is still (distance_gained * 100 / total_length).

Usage:
  python scripts/interpolate_reward_trajectory.py /path/to/reward_<MAP_NAME>.pkl [--factor 10] [--out path] [--dry-run]

Example (TmrlData on Windows WSL):
  python scripts/interpolate_reward_trajectory.py /mnt/c/Users/szulc/TmrlData/reward/reward_test-3.pkl --factor 10
"""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np


def _cumulative_distances(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length along the polyline (length at each point index)."""
    if len(points) < 2:
        return np.zeros(max(1, len(points)))
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    out = np.zeros(len(points))
    np.cumsum(diffs, out=out[1:])
    return out


def interpolate_trajectory(points: np.ndarray, factor: int) -> np.ndarray:
    """
    Return a new point array with roughly factor*len(points) points, uniformly spaced by arc length.
    Linear interpolation along the original polyline; total arc length is preserved.
    """
    n = len(points)
    if n < 2:
        return points.copy()
    cum = _cumulative_distances(points)
    total = float(cum[-1])
    if total <= 0:
        return points.copy()

    num_new = max(2, int(factor * n))
    # Uniform spacing in arc length
    s_values = np.linspace(0.0, total, num_new, endpoint=True)

    new_pts = []
    j = 0
    for s in s_values:
        if s >= total:
            new_pts.append(points[-1].copy())
            continue
        while j + 1 < n and cum[j + 1] < s:
            j += 1
        if j + 1 >= n:
            new_pts.append(points[-1].copy())
            continue
        seg_start = cum[j]
        seg_end = cum[j + 1]
        if seg_end <= seg_start:
            t = 0.0
        else:
            t = (s - seg_start) / (seg_end - seg_start)
        t = np.clip(t, 0.0, 1.0)
        pt = (1.0 - t) * points[j] + t * points[j + 1]
        new_pts.append(pt)
    return np.array(new_pts, dtype=points.dtype)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interpolate reward trajectory pkl to add more points (e.g. 10x)."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to reward_<MAP_NAME>.pkl (e.g. TmrlData/reward/reward_test-3.pkl)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=10,
        metavar="N",
        help="Target number of points = N * original length (default: 10)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: overwrite input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print stats, do not write file",
    )
    args = parser.parse_args()

    in_path = args.input
    try:
        with open(in_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {in_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading {in_path}: {e}", file=sys.stderr)
        return 1

    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 3:
        print(
            f"Error: expected (N, 3) array, got shape {getattr(data, 'shape', '?')}",
            file=sys.stderr,
        )
        return 1

    n_old = len(data)
    cum_old = _cumulative_distances(data)
    total_old = float(cum_old[-1]) if n_old >= 2 else 0.0
    avg_dist_old = total_old / (n_old - 1) if n_old > 1 else 0.0

    new_data = interpolate_trajectory(data, args.factor)
    n_new = len(new_data)
    cum_new = _cumulative_distances(new_data)
    total_new = float(cum_new[-1]) if n_new >= 2 else 0.0
    avg_dist_new = total_new / (n_new - 1) if n_new > 1 else 0.0

    # Reward scale in tmrl: raw progress per step = distance_gained * (100 / total_length).
    # Total length unchanged => total raw reward for a full lap still ~100; no explosion.
    length_ratio = total_new / total_old if total_old > 0 else 1.0

    print("Before:")
    print(f"  points: {n_old},  total length: {total_old:.2f},  avg segment: {avg_dist_old:.4f}")
    print("After:")
    print(f"  points: {n_new},  total length: {total_new:.2f},  avg segment: {avg_dist_new:.4f}")
    print(f"  length ratio (after/before): {length_ratio:.6f} (should be ~1.0)")
    if abs(length_ratio - 1.0) > 0.01:
        print("  Warning: total length changed by >1% — check interpolation.")
    else:
        print("  Reward scale unchanged (total lap raw reward still ~100).")

    if args.dry_run:
        print("(dry-run: not writing file)")
        return 0

    out_path = args.out if args.out else in_path
    try:
        with open(out_path, "wb") as f:
            pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error writing {out_path}: {e}", file=sys.stderr)
        return 1
    print(f"Saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
