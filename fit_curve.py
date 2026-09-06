import math
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).with_name("xy_data.csv")


def residual_for(params, xy):
    theta, m, x_shift = params
    c = math.cos(theta)
    s = math.sin(theta)
    x = xy[:, 0]
    y = xy[:, 1]
    t = (x - x_shift) * c + (y - 42.0) * s
    w = -(x - x_shift) * s + (y - 42.0) * c
    predicted_w = np.exp(m * np.abs(t)) * np.sin(0.3 * t)
    residual = w - predicted_w
    return float(np.mean(residual * residual)), t, w


def coordinate_descent(start, xy, steps):
    theta, m, x_shift = start
    best_score = residual_for(start, xy)[0]

    for step_theta, step_m, step_x in steps:
        improved = True
        while improved:
            improved = False
            candidates = []

            for dt in (-step_theta, 0.0, step_theta):
                for dm in (-step_m, 0.0, step_m):
                    for dx in (-step_x, 0.0, step_x):
                        candidate = (
                            min(max(theta + dt, math.radians(1e-8)), math.radians(50.0 - 1e-8)),
                            min(max(m + dm, -0.05 + 1e-12), 0.05 - 1e-12),
                            min(max(x_shift + dx, 1e-12), 100.0 - 1e-12),
                        )
                        candidates.append((residual_for(candidate, xy)[0], candidate))

            candidate_score, candidate_params = min(candidates, key=lambda item: item[0])
            if candidate_score + 1e-18 < best_score:
                best_score = candidate_score
                theta, m, x_shift = candidate_params
                improved = True

    return best_score, (theta, m, x_shift)


def fit_parameters(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    coarse = []

    # In the rotated frame, a = x*cos(theta) + (y-42)*sin(theta)
    # equals t + X*cos(theta). Since t is known to be in roughly [6, 60],
    # this gives a strong initial estimate for X for each theta.
    for theta_deg in np.linspace(0.1, 49.9, 2000):
        theta = math.radians(float(theta_deg))
        c = math.cos(theta)
        s = math.sin(theta)
        a = x * c + (y - 42.0) * s
        x_shift = 0.5 * ((float(np.min(a)) - 6.0) / c + (float(np.max(a)) - 60.0) / c)

        if not (0.0 < x_shift < 100.0):
            continue

        for m in np.linspace(-0.05, 0.05, 251):
            score = residual_for((theta, float(m), x_shift), xy)[0]
            coarse.append((score, theta, float(m), x_shift))

    coarse.sort(key=lambda row: row[0])

    steps = [
        (math.radians(0.05), 0.00005, 0.05),
        (math.radians(0.01), 0.00001, 0.01),
        (math.radians(0.002), 0.000002, 0.002),
        (math.radians(0.0004), 0.0000004, 0.0004),
        (math.radians(0.00008), 0.00000008, 0.00008),
    ]

    refined = [
        coordinate_descent((theta, m, x_shift), xy, steps)
        for _, theta, m, x_shift in coarse[:20]
    ]
    refined.sort(key=lambda row: row[0])
    return refined[0]


def main():
    df = pd.read_csv(DATA_PATH)
    xy = df[["x", "y"]].to_numpy(dtype=float)

    score, params = fit_parameters(xy)
    theta, m, x_shift = params

    reconstruction_score, t, w = residual_for(params, xy)
    abs_residual = np.abs(w - np.exp(m * np.abs(t)) * np.sin(0.3 * t))

    print(f"rows: {len(xy)}")
    print(f"theta_rad: {theta:.12f}")
    print(f"theta_deg: {math.degrees(theta):.12f}")
    print(f"M: {m:.12f}")
    print(f"X: {x_shift:.12f}")
    print(f"mean_square_residual_transformed: {reconstruction_score:.12e}")
    print(f"mean_abs_residual_transformed: {float(np.mean(abs_residual)):.12e}")
    print(f"max_abs_residual_transformed: {float(np.max(abs_residual)):.12e}")
    print(f"t_min: {float(np.min(t)):.12f}")
    print(f"t_max: {float(np.max(t)):.12f}")
    print()
    print("Desmos/LaTeX:")
    print(
        "\\left("
        f"t\\cdot\\cos({theta:.9f})-e^{{{m:.9f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\sin({theta:.9f})+{x_shift:.9f},"
        f"42+t\\cdot\\sin({theta:.9f})+e^{{{m:.9f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\cos({theta:.9f})"
        "\\right)"
    )

    # Exact rounded result used for final submission.
    theta_exact = math.pi / 6.0
    m_exact = 0.03
    x_exact = 55.0
    c = math.cos(theta_exact)
    s = math.sin(theta_exact)
    x = xy[:, 0]
    y = xy[:, 1]
    t_exact = (x - x_exact) * c + (y - 42.0) * s
    xr = t_exact * c - np.exp(m_exact * np.abs(t_exact)) * np.sin(0.3 * t_exact) * s + x_exact
    yr = 42.0 + t_exact * s + np.exp(m_exact * np.abs(t_exact)) * np.sin(0.3 * t_exact) * c
    l1 = np.abs(x - xr) + np.abs(y - yr)
    print()
    print("Rounded final-parameter xy L1 check:")
    print(f"mean_xy_L1: {float(np.mean(l1)):.12e}")
    print(f"max_xy_L1: {float(np.max(l1)):.12e}")
    print(f"total_xy_L1: {float(np.sum(l1)):.12e}")


if __name__ == "__main__":
    main()
