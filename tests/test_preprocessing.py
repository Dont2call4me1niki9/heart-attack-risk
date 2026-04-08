import pandas as pd

from src.config import AppConfig
from src.preprocessing import Preprocessor


def test_preprocessor_fit_transform() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "id": [100, 101],
            "Gender": ["1.0", "Female"],
            "Age": [45, None],
            "Heart Attack Risk (Binary)": [1, 0],
        }
    )

    preprocessor = Preprocessor(config=AppConfig())
    result = preprocessor.fit_transform(df.drop(columns=["Heart Attack Risk (Binary)"]))

    assert "id" not in result.columns
    assert "Unnamed: 0" not in result.columns
    assert "Gender" in result.columns
    assert result["Gender"].iloc[0] == "Male"
    assert result["Age"].isna().sum() == 0
