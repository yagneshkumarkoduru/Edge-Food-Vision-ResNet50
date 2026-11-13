"""
Measured validation of the conformal prediction and C-TTA engines on synthetic
probability streams. All reported numbers in this module are produced by
running the calibration and update code in this repository, not by hardcoded
constants. Requires numpy only; no torch dependency.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from implementations.v2_conformal_risk_calibration.conformal_prediction_engine import (
    SplitConformalCalibrator,
)
from implementations.v3_dynamic_exit_test_time_adaptation.online_conformal_tta_tracker import (
    OnlineConformalTTATracker,
)

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def make_stream(num_steps, num_classes=10, seed=42, drift_profile="two_phase"):
    """
    Synthetic softmax stream with controlled covariate shift.
    drift in [0, 1]; higher drift lowers the true-class probability boost,
    simulating glare/blur degrading classifier confidence.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=num_steps)
    probs = np.zeros((num_steps, num_classes))
    drifts = np.zeros(num_steps)

    for t in range(num_steps):
        if drift_profile == "clean":
            drift = 0.0
        elif drift_profile == "two_phase":
            if t < num_steps // 4:
                drift = 0.0
            elif t < 3 * num_steps // 4:
                drift = 0.55 * (t - num_steps // 4) / (num_steps // 2)
            else:
                drift = 0.55 + 0.35 * (t - 3 * num_steps // 4) / (num_steps // 4)
        else:
            raise ValueError(f"unknown drift_profile: {drift_profile}")

        p = rng.dirichlet(np.ones(num_classes) * 0.7)
        boost = 2.2 * max(0.0, 1.0 - drift)
        p[labels[t]] += boost
        p /= p.sum()
        probs[t] = p
        drifts[t] = drift

    return probs, labels, drifts


def measure_static_coverage(alpha=0.05, num_calib=400, num_test=600, seed=42):
    """Split-conformal coverage on exchangeable (clean) synthetic data."""
    probs, labels, _ = make_stream(num_calib + num_test, seed=seed, drift_profile="clean")
    calib_scores = 1.0 - probs[np.arange(num_calib), labels[:num_calib]]

    cal = SplitConformalCalibrator(alpha=alpha)
    cal.calibrate(calib_scores)

    covered = 0
    set_sizes = []
    for j in range(num_calib, num_calib + num_test):
        pred_set = cal.predict_prediction_set(probs[j])
        set_sizes.append(len(pred_set))
        if labels[j] in pred_set:
            covered += 1

    return {
        "alpha": alpha,
        "target_coverage_pct": (1.0 - alpha) * 100.0,
        "measured_coverage_pct": covered / num_test * 100.0,
        "mean_set_size": float(np.mean(set_sizes)),
        "q_hat": cal.q_hat,
        "num_calib": num_calib,
        "num_test": num_test,
        "seed": seed,
    }


def measure_static_vs_ctta_under_drift(alpha=0.05, num_steps=1000, num_calib=150,
                                       gamma=0.01, seed=42):
    """
    Static split-conformal (calibrated on initial clean segment) versus online
    C-TTA pinball update on a drifting stream. Returns realized coverages.
    """
    probs, labels, drifts = make_stream(num_steps, seed=seed, drift_profile="two_phase")

    calib_scores = 1.0 - probs[np.arange(num_calib), labels[:num_calib]]
    cal = SplitConformalCalibrator(alpha=alpha)
    q_static = cal.calibrate(calib_scores)

    scores = 1.0 - probs[np.arange(num_steps), labels]

    static_covered = scores <= q_static

    tracker = OnlineConformalTTATracker(alpha=alpha, step_size_gamma=gamma, q_init=q_static)
    online_covered = np.zeros(num_steps, dtype=bool)
    q_trace = np.zeros(num_steps)
    for t in range(num_steps):
        cov = bool(scores[t] <= tracker.q_t)
        online_covered[t] = cov
        q_trace[t] = tracker.q_t
        tracker.update_online(cov)

    eval_slice = slice(num_calib, None)
    n_eval = num_steps - num_calib

    def windowed_cov(mask, start, end):
        return float(mask[start:end].mean() * 100.0)

    return {
        "alpha": alpha,
        "target_coverage_pct": (1.0 - alpha) * 100.0,
        "num_steps": num_steps,
        "num_calib": num_calib,
        "gamma": gamma,
        "seed": seed,
        "q_static": q_static,
        "q_final_online": tracker.q_t,
        "static_overall_pct": float(static_covered[eval_slice].mean() * 100.0),
        "online_overall_pct": float(online_covered[eval_slice].mean() * 100.0),
        "online_mean_q_t": float(q_trace[eval_slice].mean()),
        "static_clean_window_pct": windowed_cov(static_covered, num_calib, num_steps // 4),
        "static_drift_window_pct": windowed_cov(static_covered, 3 * num_steps // 4, num_steps),
        "online_clean_window_pct": windowed_cov(online_covered, num_calib, num_steps // 4),
        "online_drift_window_pct": windowed_cov(online_covered, 3 * num_steps // 4, num_steps),
        "online_final_200_pct": float(online_covered[-200:].mean() * 100.0),
        "max_drift": float(drifts.max()),
        "n_eval": n_eval,
    }


def measure_early_exit_flops(num_samples=2000, tau_exit=1.55, seed=42):
    """
    Runs the repository's DynamicEarlyExitNetwork over random features and
    measures the realized exit distribution and FLOP reduction. The per-exit
    FLOP fractions (0.32 / 0.65 / 1.0) are model assumptions; the measured
    quantity is the exit distribution they produce.
    """
    from implementations.v3_dynamic_exit_test_time_adaptation.dynamic_early_exit_resnet50 import (
        DynamicEarlyExitNetwork,
    )

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    net = DynamicEarlyExitNetwork(tau_exit=tau_exit)

    exit_counts = {}
    flop_fracs = []
    for _ in range(num_samples):
        res = net.infer_with_early_exit(rng.standard_normal(2048))
        exit_counts[res["exit_taken"]] = exit_counts.get(res["exit_taken"], 0) + 1
        flop_fracs.append(res["flop_fraction"])

    mean_flops = float(np.mean(flop_fracs))
    return {
        "num_samples": num_samples,
        "tau_exit": tau_exit,
        "seed": seed,
        "exit_distribution_pct": {k: v / num_samples * 100.0 for k, v in exit_counts.items()},
        "measured_flop_reduction_pct": (1.0 - mean_flops) * 100.0,
        "note": "Exit distribution measured from the repository's simulator; "
                "per-exit FLOP fractions 0.32/0.65/1.0 are model assumptions, "
                "not measured on hardware.",
    }


def run_all(out_path=None):
    results = {
        "evidence_class": "MEASURED (synthetic data, numpy-only)",
        "static_clean": measure_static_coverage(),
        "static_vs_ctta_drift": measure_static_vs_ctta_under_drift(),
        "early_exit": measure_early_exit_flops(),
    }
    if out_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "conformal_measurement.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results, out_path


if __name__ == "__main__":
    results, out_path = run_all()
    sc = results["static_clean"]
    dr = results["static_vs_ctta_drift"]
    ee = results["early_exit"]
    print(f"Static conformal (clean): target {sc['target_coverage_pct']:.1f}%, "
          f"measured {sc['measured_coverage_pct']:.2f}%")
    print(f"Under drift: static {dr['static_overall_pct']:.2f}% overall, "
          f"{dr['static_drift_window_pct']:.2f}% in severe window")
    print(f"C-TTA:       {dr['online_overall_pct']:.2f}% overall, "
          f"{dr['online_drift_window_pct']:.2f}% in severe window, "
          f"final-200 {dr['online_final_200_pct']:.2f}%")
    print(f"Early exit FLOP reduction (simulated): {ee['measured_flop_reduction_pct']:.2f}%")
    print(f"Results written to {out_path}")
