from dataclasses import dataclass

import pandas as pd


@dataclass
class DataLoader:
    train_path: str | None = None
    test_path: str | None = None

    def load_train(self) -> pd.DataFrame:
        if not self.train_path:
            raise ValueError("Train path is not provided.")
        return pd.read_csv(self.train_path)

    def load_test(self) -> pd.DataFrame:
        if not self.test_path:
            raise ValueError("Test path is not provided.")
        return pd.read_csv(self.test_path)
