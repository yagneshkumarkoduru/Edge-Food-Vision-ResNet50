"""
Tests for the V2 static split-conformal engine and the V3 online C-TTA
tracker. All tests run on synthetic numpy data; no torch, no Food-101.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from implementations.v2_conformal_risk_calibration.conformal_prediction_engine import (
    SplitConformalCalibrator,
)
from implementations.v3_dynamic_exit_test_time_adaptation.online_conformal_tta_tracker import (
    OnlineConformalTTATracker,
)
from src.validate_conformal_engine import (
    make_stream,
    run_static_clean,
    run_drift_comparison,
)


class TestSplitConformalCalibrator:
    def test_quantile_level_formula(self):
        """q_hat uses the finite-sample level ceil((n+1)(1-alpha))/n."""
        cal = SplitConformalCalibrator(alpha=0.05)
        scores = np.linspace(0.0, 1.0, 100)
        q = cal.calibrate(scores)
        n = 100
        level = np.ceil((n + 1) * 0.95) / n
        expected = float(np.quantile(scores, level, method="higher"))
        assert q == pytest.approx(expected)
        assert cal.q_hat == pytest.approx(expected)

    def test_prediction_set_contains_high_prob_classes(self):
        cal = SplitConformalCalibrator(alpha=0.05)
        cal.calibrate(np.full(200, 0.3))  # q_hat = 0.3 -> threshold 0.7
        probs = np.array([0.8, 0.1, 0.05, 0.05])
        pred = cal.predict_prediction_set(probs)
        assert pred == [0]

    def test_prediction_set_never_empty(self):
        """Falls back to argmax when no class clears the threshold."""
        cal = SplitConformalCalibrator(alpha=0.05)
        cal.calibrate(np.zeros(50))  # q_hat = 0 -> threshold 1.0
        probs = np.array([0.4, 0.35, 0.25])
        pred = cal.predict_prediction_set(probs)
        assert pred == [0]  # argmax fallback

    def test_uncalibrated_raises(self):
        cal = SplitConformalCalibrator(alpha=0.05)
        with pytest.raises(ValueError):
            cal.predict_prediction_set(np.array([0.5, 0.5]))

    def test_marginal_coverage_on_clean_stream(self):
        """Measured coverage on exchangeable data meets the 1-alpha target."""
        res = run_static_clean(alpha=0.05, num_calib=400, num_test=600, seed=7)
        assert res["measured_coverage_pct"] >= res["target_coverage_pct"] - 2.0


class TestOnlineConformalTTATracker:
    def test_update_direction(self):
        """Coverage raises q slightly; a miss lowers q by (1-alpha)*gamma."""
        tr = OnlineConformalTTATracker(alpha=0.05, step_size_gamma=0.01,
                                       q_init=0.5)
        tr.update_online(True)
        assert tr.q_t == pytest.approx(0.5 + 0.01 * 0.05)
        tr.update_online(False)
        assert tr.q_t == pytest.approx(0.5 + 0.01 * 0.05 - 0.01 * 0.95)

    def test_q_clipped_to_bounds(self):
        tr = OnlineConformalTTATracker(alpha=0.05, step_size_gamma=0.01,
                                       q_init=0.99)
        for _ in range(500):
            tr.update_online(True)
        assert tr.q_t <= 0.99
        tr2 = OnlineConformalTTATracker(alpha=0.05, step_size_gamma=0.01,
                                        q_init=0.05)
        for _ in range(500):
            tr2.update_online(False)
        assert tr2.q_t >= 0.05

    def test_history_records_every_step(self):
        tr = OnlineConformalTTATracker(alpha=0.05, step_size_gamma=0.01)
        for i in range(25):
            tr.update_online(i % 2 == 0)
        assert len(tr.history) == 25


class TestStreamValidation:
    def test_make_stream_shapes_and_range(self):
        probs, labels, drifts = make_stream(300, seed=1, drift_profile="two_phase")
        assert probs.shape == (300, 10)
        assert labels.shape == (300,)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert drifts[0] == 0.0 and drifts[-1] > 0.5

    def test_static_coverage_collapses_under_drift(self):
        """Static calibration on the clean segment loses coverage in the
        severe-drift window by a wide margin."""
        res = run_drift_comparison(seed=42)
        assert (res["static_clean_window_pct"]
                > res["static_drift_window_pct"] + 40.0)

    def test_ctta_recovers_coverage_above_static(self):
        """At gamma=0.01 the online tracker beats static coverage overall."""
        res = run_drift_comparison(seed=42, gamma=0.01)
        assert res["online_overall_pct"] > res["static_overall_pct"] + 20.0

    def test_ctta_gamma_sensitivity_is_real(self):
        """Documents the measured gamma sensitivity: too-small gamma collapses
        the tracker to its q floor on this drift profile."""
        low = run_drift_comparison(seed=42, gamma=0.002)
        mid = run_drift_comparison(seed=42, gamma=0.01)
        assert low["online_overall_pct"] < mid["online_overall_pct"]

    def test_validation_artifact_written(self, tmp_path):
        from src.validate_conformal_engine import run_all
        out = tmp_path / "artifact.json"
        results, path = run_all(out_path=str(out))
        assert os.path.exists(path)
        assert results["static_clean"]["measured_coverage_pct"] > 90.0
        assert "headline_claim_audit" in results
