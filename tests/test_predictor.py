import pandas as pd
import pytest

from src.config import AppConfig
from src.predictor import HeartRiskPredictor


def test_predict_without_loaded_model_raises_error() -> None:
    predictor = HeartRiskPredictor(config=AppConfig())
    df = pd.DataFrame({"id": [1], "Gender": ["Male"], "Age": [50]})

    with pytest.raises(ValueError):
        predictor.predict(df)
