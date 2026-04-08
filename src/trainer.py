from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from src.config import AppConfig
from src.metrics import MetricsEvaluator
from src.preprocessing import Preprocessor
from src.utils import FileUtils


@dataclass
class ModelTrainer:
    config: AppConfig
    preprocessor: Preprocessor
    metrics_evaluator: MetricsEvaluator
    model: CatBoostClassifier | None = None

    def _build_model(self) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=self.config.random_state,
            verbose=False,
        )

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found.")

        y = df[self.config.target_column].astype(int)
        X = df.drop(columns=[self.config.target_column])

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        X_train_prepared = self.preprocessor.fit_transform(X_train)
        X_valid_prepared = self.preprocessor.transform(X_valid)
        cat_features = self.preprocessor.get_cat_feature_indices()

        self.model = self._build_model()
        self.model.fit(
            X_train_prepared,
            y_train,
            cat_features=cat_features,
            eval_set=(X_valid_prepared, y_valid),
            use_best_model=True,
        )

        valid_proba = self.model.predict_proba(X_valid_prepared)[:, 1]
        threshold_info = self.metrics_evaluator.find_best_threshold(y_valid, valid_proba)
        threshold = threshold_info["best_threshold"]
        valid_pred = (valid_proba >= threshold).astype(int)

        metrics = self.metrics_evaluator.calculate_classification_metrics(
            y_true=y_valid,
            y_pred=valid_pred,
            y_proba=valid_proba,
        )

        return {
            "metrics": metrics,
            "threshold": threshold,
            "feature_columns": self.preprocessor.feature_columns_,
            "categorical_columns": self.config.categorical_columns,
        }

    def fit_on_full_data(self, df: pd.DataFrame) -> None:
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found.")

        y = df[self.config.target_column].astype(int)
        X = df.drop(columns=[self.config.target_column])

        X_prepared = self.preprocessor.fit_transform(X)
        cat_features = self.preprocessor.get_cat_feature_indices()

        self.model = self._build_model()
        self.model.fit(X_prepared, y, cat_features=cat_features)

    def save_model(self, model_path: str, metadata_path: str, threshold: float) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        FileUtils.ensure_parent_dir(model_path)
        self.model.save_model(model_path)

        metadata = {
            "threshold": threshold,
            "feature_columns": self.preprocessor.feature_columns_,
            "numeric_fill_values": self.preprocessor.numeric_fill_values_,
            "categorical_fill_values": self.preprocessor.categorical_fill_values_,
            "categorical_columns": self.config.categorical_columns,
            "id_column": self.config.id_column,
            "drop_columns": self.config.drop_columns,
            "target_column": self.config.target_column,
        }
        FileUtils.save_json(metadata, metadata_path)
