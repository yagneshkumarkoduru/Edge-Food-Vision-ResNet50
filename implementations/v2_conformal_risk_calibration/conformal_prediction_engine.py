"""
Split-Conformal Prediction Engine for Provable Safety Coverage.
Guarantees marginal coverage: P(Y in C(X)) >= 1 - alpha
using non-conformity scores s_i = 1 - f(X_i)_{Y_i} and finite-sample quantile calibration:
q_hat = Quantile_{ceil((n+1)(1-alpha))/n}(s_1, ..., s_n).
"""

import numpy as np


class SplitConformalCalibrator:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, calibration_scores: np.ndarray) -> float:
        """
        Computes the conformal quantile threshold:
        p = ceil((n + 1) * (1 - alpha)) / n
        """
        n = len(calibration_scores)
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(max(level, 0.0), 1.0)
        self.q_hat = float(np.quantile(calibration_scores, level, method="higher"))
        return self.q_hat

    def predict_prediction_set(self, probabilities: np.ndarray) -> list:
        """
        Constructs prediction set C(x) = {y : 1 - p(y|x) <= q_hat} = {y : p(y|x) >= 1 - q_hat}
        """
        if self.q_hat is None:
            raise ValueError("Calibrator not calibrated yet.")
        threshold = 1.0 - self.q_hat
        included_indices = np.where(probabilities >= threshold)[0]
        if len(included_indices) == 0:
            # Fallback to argmax to avoid empty sets
            included_indices = np.array([np.argmax(probabilities)])
        return included_indices.tolist()
