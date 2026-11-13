"""
Validation harness for the conformal prediction engine (V2 static split-conformal
and V3 online C-TTA). Runs both engines on synthetic classification score streams
with controlled distribution shift and records every realized number to a JSON
artifact. Nothing in this file is hardcoded from the headline claims; every
reported value is produced by executing the engines in this repository.

Synthetic data only: no Food-101 images, no torch, numpy only.
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
ARTIFACT_PATH = os.path.join(RESULTS_DIR, "conformal_validation.json")


def make_stream(num_steps, num_classes=10, seed=0, drift_profile="clean",
                max_drift=0.55):
    """
    Synthetic softmax probability stream with a controlled covariate-shift
    schedule. drift in [0, 1]; the true-class logit boost is scaled by
    (1 - drift), so higher drift degrades classifier confidence on the true
    label and raises the non-conformity score s = 1 - p_true.

    drift_profile:
      clean     - drift pinned at 0 for the whole stream
      ramp      - drift ramps linearly from 0 to max_drift over the stream
      two_phase - clean first quarter, ramp to max_drift by three quarters,
                  then a steeper ramp to ~1.6 * max_drift (severe tail)
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=num_steps)
    probs = np.zeros((num_steps, num_classes))
    drifts = np.zeros(num_steps)

    for t in range(num_steps):
        if drift_profile == "clean":
            drift = 0.0
        elif drift_profile == "ramp":
            drift = max_drift * t / max(num_steps - 1, 1)
        elif drift_profile == "two_phase":
            q1 = num_steps // 4
            q3 = 3 * num_steps // 4
            if t < q1:
                drift = 0.0
            elif t < q3:
                drift = max_drift * (t - q1) / (q3 - q1)
            else:
                drift = max_drift + 0.6 * max_drift * (t - q3) / max(num_steps - q3, 1)
        else:
            raise ValueError(f"unknown drift_profile: {drift_profile}")

        p = rng.dirichlet(np.ones(num_classes) * 0.7)
        p[labels[t]] += 2.2 * max(0.0, 1.0 - drift)
        p /= p.sum()
        probs[t] = p
        drifts[t] = drift

    return probs, labels, drifts


def run_static_clean(alpha=0.05, num_calib=400, num_test=600, seed=42):
    """Static split-conformal on exchangeable data; measures realized coverage."""
    probs, labels, _ = make_stream(num_calib + num_test, seed=seed,
                                   drift_profile="clean")
    calib_scores = 1.0 - probs[np.arange(num_calib), labels[:num_calib]]

    cal = SplitConformalCalibrator(alpha=alpha)
    q_hat = cal.calibrate(calib_scores)

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
        "q_hat": q_hat,
        "num_calib": num_calib,
        "num_test": num_test,
        "seed": seed,
    }


def run_drift_comparison(alpha=0.05, num_steps=1000, num_calib=150, gamma=0.01,
                         seed=42, drift_profile="two_phase", max_drift=0.55):
    """
    Static split-conformal (calibrated on the initial clean segment) versus the
    online C-TTA pinball tracker on a drifting stream. Measures realized
    coverage of both engines over the same stream, plus the realized mean
    prediction-set size of the C-TTA tracker (conservatism cost).
    """
    probs, labels, drifts = make_stream(num_steps, seed=seed,
                                        drift_profile=drift_profile,
                                        max_drift=max_drift)

    calib_scores = 1.0 - probs[np.arange(num_calib), labels[:num_calib]]
    cal = SplitConformalCalibrator(alpha=alpha)
    q_static = cal.calibrate(calib_scores)

    scores = 1.0 - probs[np.arange(num_steps), labels]
    static_covered = scores <= q_static

    tracker = OnlineConformalTTATracker(alpha=alpha, step_size_gamma=gamma,
                                        q_init=q_static)
    online_covered = np.zeros(num_steps, dtype=bool)
    q_trace = np.zeros(num_steps)
    set_sizes = np.zeros(num_steps)
    for t in range(num_steps):
        threshold = 1.0 - tracker.q_t
        online_covered[t] = bool(scores[t] <= tracker.q_t)
        q_trace[t] = tracker.q_t
        set_sizes[t] = int(np.sum(probs[t] >= threshold))
        tracker.update_online(bool(online_covered[t]))

    eval_slice = slice(num_calib, None)
    q1 = num_steps // 4
    q3 = 3 * num_steps // 4

    def window(mask, a, b):
        return float(mask[a:b].mean() * 100.0)

    return {
        "alpha": alpha,
        "target_coverage_pct": (1.0 - alpha) * 100.0,
        "num_steps": num_steps,
        "num_calib": num_calib,
        "gamma": gamma,
        "seed": seed,
        "drift_profile": drift_profile,
        "max_drift_param": max_drift,
        "realized_max_drift": float(drifts.max()),
        "q_static": q_static,
        "q_final_online": float(tracker.q_t),
        "q_online_max": float(q_trace.max()),
        "static_overall_pct": float(static_covered[eval_slice].mean() * 100.0),
        "online_overall_pct": float(online_covered[eval_slice].mean() * 100.0),
        "static_clean_window_pct": window(static_covered, num_calib, q1),
        "static_drift_window_pct": window(static_covered, q3, num_steps),
        "online_clean_window_pct": window(online_covered, num_calib, q1),
        "online_drift_window_pct": window(online_covered, q3, num_steps),
        "online_final_200_pct": float(online_covered[-200:].mean() * 100.0),
        "online_mean_set_size": float(set_sizes[eval_slice].mean()),
        "online_mean_set_size_drift_window": float(set_sizes[q3:].mean()),
        "static_mean_set_size_clean": float(
            np.mean([np.sum(probs[t] >= 1.0 - q_static)
                     for t in range(num_calib, q1)])
        ),
    }


def run_seed_sweep(seeds=(0, 1, 2, 3, 4), **kwargs):
    """Repeats the drift comparison across seeds and reports spread."""
    runs = [run_drift_comparison(seed=s, **kwargs) for s in seeds]
    static_dw = np.array([r["static_drift_window_pct"] for r in runs])
    online_dw = np.array([r["online_drift_window_pct"] for r in runs])
    online_ov = np.array([r["online_overall_pct"] for r in runs])
    set_size = np.array([r["online_mean_set_size_drift_window"] for r in runs])
    return {
        "seeds": list(seeds),
        "static_drift_window_pct_mean": float(static_dw.mean()),
        "static_drift_window_pct_std": float(static_dw.std()),
        "online_drift_window_pct_mean": float(online_dw.mean()),
        "online_drift_window_pct_std": float(online_dw.std()),
        "online_overall_pct_mean": float(online_ov.mean()),
        "online_overall_pct_std": float(online_ov.std()),
        "online_set_size_drift_mean": float(set_size.mean()),
        "per_seed": [
            {
                "seed": r["seed"],
                "static_drift_window_pct": r["static_drift_window_pct"],
                "online_drift_window_pct": r["online_drift_window_pct"],
                "online_overall_pct": r["online_overall_pct"],
                "online_mean_set_size_drift_window":
                    r["online_mean_set_size_drift_window"],
                "q_final_online": r["q_final_online"],
            }
            for r in runs
        ],
    }


def run_gamma_sweep(gammas=(0.002, 0.005, 0.01, 0.02), **kwargs):
    """C-TTA sensitivity to the step size gamma on the same drift stream."""
    out = []
    for g in gammas:
        r = run_drift_comparison(gamma=g, **kwargs)
        out.append({
            "gamma": g,
            "online_overall_pct": r["online_overall_pct"],
            "online_drift_window_pct": r["online_drift_window_pct"],
            "online_mean_set_size_drift_window":
                r["online_mean_set_size_drift_window"],
            "q_final_online": r["q_final_online"],
        })
    return out


def run_all(out_path=ARTIFACT_PATH):
    results = {
        "evidence_class": "MEASURED (synthetic numpy data; engines executed in repo)",
        "generator": "src/validate_conformal_engine.py",
        "static_clean": run_static_clean(),
        "drift_comparison_seed42": run_drift_comparison(),
        "seed_sweep": run_seed_sweep(),
        "gamma_sweep": run_gamma_sweep(seed=42),
        "headline_claim_audit": {
            "94.50_pct_coverage": {
                "status": "NOT REPRODUCED",
                "note": "Hardcoded literal in docs/paper and README. Measured "
                        "online C-TTA coverage on the synthetic drift stream at "
                        "gamma=0.01 is 100 pct, overshooting the 95 pct target; "
                        "the realized coverage is highly sensitive to gamma "
                        "(collapse to the q floor at gamma<=0.005, saturation "
                        "to the q ceiling at gamma=0.02) and never lands at "
                        "the claimed 94.50 pct.",
            },
            "49.79_pct_flop_reduction": {
                "status": "MODEL ASSUMPTION",
                "note": "Per-exit FLOP fractions 0.32/0.65/1.0 are assumptions "
                        "in dynamic_early_exit_resnet50.py; no hardware or "
                        "profiler measurement exists in this repository.",
            },
            "87.5_pct_size_reduction": {
                "status": "HARDCODED LITERAL",
                "note": "97.8 to 12.2 MB figures are literals in "
                        "pruning_aware_quantization.py; no serialized checkpoint "
                        "is measured.",
            },
            "4.71x_speedup": {
                "status": "HARDCODED LITERAL",
                "note": "Latency figures are literals in "
                        "pruning_aware_quantization.py; no timing harness exists.",
            },
            "40.00_pct_static_collapse": {
                "status": "NOT REPRODUCED AS STATED",
                "note": "Measured static coverage collapse is real but its "
                        "magnitude depends on the drift schedule; on the "
                        "two_phase stream the severe-window coverage is far "
                        "below 40 pct, not exactly 40.00 pct.",
            },
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results, out_path


if __name__ == "__main__":
    results, out_path = run_all()
    sc = results["static_clean"]
    dr = results["drift_comparison_seed42"]
    sw = results["seed_sweep"]
    print(f"Static conformal (clean):  target {sc['target_coverage_pct']:.1f}%, "
          f"measured {sc['measured_coverage_pct']:.2f}%")
    print(f"Drift stream (seed 42):    static overall {dr['static_overall_pct']:.2f}%, "
          f"severe-window {dr['static_drift_window_pct']:.2f}%")
    print(f"C-TTA online (seed 42):    overall {dr['online_overall_pct']:.2f}%, "
          f"severe-window {dr['online_drift_window_pct']:.2f}%, "
          f"mean set size (drift) {dr['online_mean_set_size_drift_window']:.2f}/10")
    print(f"Seed sweep:                static severe {sw['static_drift_window_pct_mean']:.2f}"
          f" +/- {sw['static_drift_window_pct_std']:.2f}%, "
          f"online severe {sw['online_drift_window_pct_mean']:.2f}"
          f" +/- {sw['online_drift_window_pct_std']:.2f}%")
    print(f"Artifact written to {out_path}")
