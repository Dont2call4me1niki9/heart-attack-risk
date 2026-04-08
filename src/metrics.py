from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class MetricsEvaluator:
    def calculate_classification_metrics(
        self,
        y_true,
        y_pred,
        y_proba,
    ) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
        }

    def find_best_threshold(self, y_true, y_proba) -> Dict[str, float]:
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in np.arange(0.1, 0.91, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(round(threshold, 2))

        return {
            "best_threshold": best_threshold,
            "best_f1": float(best_f1),
        }
