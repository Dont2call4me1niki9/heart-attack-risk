from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from catboost import CatBoostClassifier

from src.config import AppConfig
from src.preprocessing import Preprocessor
from src.utils import FileUtils


@dataclass
class HeartRiskPredictor:
    config: AppConfig
    preprocessor: Preprocessor | None = None
    model: CatBoostClassifier | None = None
    threshold: float = 0.5

    def load_model(self, model_path: str, metadata_path: str) -> None:
        metadata = FileUtils.load_json(metadata_path)

        config = AppConfig(
            target_column=metadata.get("target_column", self.config.target_column),
            id_column=metadata.get("id_column", self.config.id_column),
            drop_columns=metadata.get("drop_columns", self.config.drop_columns),
            categorical_columns=metadata.get(
                "categorical_columns", self.config.categorical_columns
            ),
            threshold=metadata.get("threshold", self.config.threshold),
        )
        self.config = config

        self.preprocessor = Preprocessor(config=self.config)
        self.preprocessor.feature_columns_ = metadata["feature_columns"]
        self.preprocessor.numeric_fill_values_ = metadata["numeric_fill_values"]
        self.preprocessor.categorical_fill_values_ = metadata["categorical_fill_values"]

        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.threshold = float(metadata.get("threshold", 0.5))

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model is not loaded.")

        prepared = self.preprocessor.transform(df)
        proba = self.model.predict_proba(prepared)[:, 1]
        return pd.Series(proba)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_proba(df)
        predictions = (probabilities >= self.threshold).astype(int)

        if self.config.id_column not in df.columns:
            raise ValueError(f"Input dataframe must contain '{self.config.id_column}'.")

        return pd.DataFrame(
            {
                "id": df[self.config.id_column].values,
                "prediction": predictions.values,
            }
        )

    def predict_from_csv(self, csv_path: str, output_path: str | None = None) -> Dict[str, str | int]:
        df = pd.read_csv(csv_path)
        result = self.predict(df)

        if output_path:
            FileUtils.ensure_parent_dir(output_path)
            result.to_csv(output_path, index=False)

        return {
            "rows": len(result),
            "output_path": output_path or "",
        }
