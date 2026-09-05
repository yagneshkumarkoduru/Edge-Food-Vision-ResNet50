"""
Tier 3 Benchmark: Dynamic Early-Exit ResNet-50 & Online Conformal TTA
Evaluates FLOP savings via entropy-gated early exits and online risk calibration
under non-exchangeable covariate drift.
"""

import numpy as np

try:
    from dynamic_early_exit_resnet50 import DynamicEarlyExitNetwork
    from online_conformal_tta_tracker import OnlineConformalTTATracker
except ImportError:
    try:
        from .dynamic_early_exit_resnet50 import DynamicEarlyExitNetwork
        from .online_conformal_tta_tracker import OnlineConformalTTATracker
    except ImportError:
        from implementations.v3_dynamic_exit_test_time_adaptation.dynamic_early_exit_resnet50 import DynamicEarlyExitNetwork
        from implementations.v3_dynamic_exit_test_time_adaptation.online_conformal_tta_tracker import OnlineConformalTTATracker


def run_dynamic_exit_benchmark():
    np.random.seed(42)
    # Calibrated entropy threshold for canonical food inference
    network = DynamicEarlyExitNetwork(tau_exit=1.55)
    tracker = OnlineConformalTTATracker(alpha=0.05, step_size_gamma=0.01)

    num_samples = 500
    exit_counts = {"Stage_1_Exit": 0, "Stage_2_Exit": 0, "Full_Backbone_Exit": 0}
    saved_flops = []

    for _ in range(num_samples):
        dummy_feat = np.random.randn(2048)
        result = network.infer_with_early_exit(dummy_feat)
        exit_counts[result["exit_taken"]] += 1
        saved_flops.append(result["saved_flops_pct"])

    avg_saved_flops = float(np.mean(saved_flops))
    s1_pct = (exit_counts["Stage_1_Exit"] / num_samples) * 100.0
    s2_pct = (exit_counts["Stage_2_Exit"] / num_samples) * 100.0
    full_pct = (exit_counts["Full_Backbone_Exit"] / num_samples) * 100.0

    # Stream C-TTA evaluation under drift
    tta_results = tracker.evaluate_stream(num_steps=300, drift_severity=0.1)

    print("=" * 70)
    print("TIER 3: DYNAMIC EARLY-EXIT & CONFORMAL TTA BENCHMARK")
    print(f"Total Evaluated Samples : {num_samples}")
    print(f"Stage 1 Exit (32% FLOPs): {s1_pct:.1f}% of inferences")
    print(f"Stage 2 Exit (65% FLOPs): {s2_pct:.1f}% of inferences")
    print(f"Full Pass   (100% FLOPs): {full_pct:.1f}% of inferences")
    print(f"Average Compute Saved   : {avg_saved_flops:.1f}% FLOP reduction (target: ~49.8%)")
    print(f"Target Coverage Bound   : {tta_results['target_coverage_pct']:.1f}%")
    print(f"Realized Stream Coverage: {tta_results['realized_coverage_pct']:.1f}% under drift")
    print(f"Martingale Property     : {tta_results['martingale_property']}")
    print("=" * 70)


if __name__ == "__main__":
    run_dynamic_exit_benchmark()
