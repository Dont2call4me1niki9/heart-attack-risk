from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.config import AppConfig


@dataclass
class Preprocessor:
    config: AppConfig
    numeric_fill_values_: Dict[str, float] = field(default_factory=dict)
    categorical_fill_values_: Dict[str, str] = field(default_factory=dict)
    feature_columns_: List[str] = field(default_factory=list)

    def _clean_gender(self, series: pd.Series) -> pd.Series:
        mapping = {
            "1": "Male",
            "1.0": "Male",
            1: "Male",
            1.0: "Male",
            "0": "Female",
            "0.0": "Female",
            0: "Female",
            0.0: "Female",
            "male": "Male",
            "female": "Female",
            "Male": "Male",
            "Female": "Female",
        }
        return series.map(lambda x: mapping.get(x, x))

    def _drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = [
            column
            for column in self.config.drop_columns + [self.config.target_column]
            if column in df.columns
        ]
        return df.drop(columns=columns_to_drop, errors="ignore")

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if self.config.id_column in result.columns:
            result = result.drop(columns=[self.config.id_column], errors="ignore")

        if "Gender" in result.columns:
            result["Gender"] = self._clean_gender(result["Gender"])
            result["Gender"] = result["Gender"].astype("object")

        result = self._drop_unused_columns(result)
        return result

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        prepared = self._normalize_dataframe(df)
        self.feature_columns_ = prepared.columns.tolist()

        for column in prepared.columns:
            if column in self.config.categorical_columns:
                mode = prepared[column].mode(dropna=True)
                self.categorical_fill_values_[column] = (
                    mode.iloc[0] if not mode.empty else "Unknown"
                )
            else:
                self.numeric_fill_values_[column] = pd.to_numeric(
                    prepared[column], errors="coerce"
                ).median()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._normalize_dataframe(df)

        for column in self.feature_columns_:
            if column not in prepared.columns:
                prepared[column] = None

        prepared = prepared[self.feature_columns_].copy()

        for column in self.feature_columns_:
            if column in self.config.categorical_columns:
                prepared[column] = prepared[column].fillna(
                    self.categorical_fill_values_.get(column, "Unknown")
                )
                prepared[column] = prepared[column].astype(str)
            else:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
                prepared[column] = prepared[column].fillna(
                    self.numeric_fill_values_.get(column, 0.0)
                )

        return prepared

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_cat_feature_indices(self) -> List[int]:
        return [
            index
            for index, column in enumerate(self.feature_columns_)
            if column in self.config.categorical_columns
        ]
