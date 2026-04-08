from dataclasses import dataclass, field
from typing import List


@dataclass
class AppConfig:
    target_column: str = "Heart Attack Risk (Binary)"
    id_column: str = "id"
    drop_columns: List[str] = field(default_factory=lambda: ["Unnamed: 0"])
    categorical_columns: List[str] = field(default_factory=lambda: ["Gender"])
    random_state: int = 42
    test_size: float = 0.2
    threshold: float = 0.5
