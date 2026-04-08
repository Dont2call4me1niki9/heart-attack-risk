import json
import os
from typing import Any, Dict


class FileUtils:
    @staticmethod
    def ensure_parent_dir(file_path: str) -> None:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        FileUtils.ensure_parent_dir(file_path)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
