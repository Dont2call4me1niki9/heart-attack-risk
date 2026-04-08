import argparse

from src.config import AppConfig
from src.predictor import HeartRiskPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict heart attack risk.")
    parser.add_argument("--test-path", required=True, help="Path to test CSV file.")
    parser.add_argument("--model-path", required=True, help="Path to model file.")
    parser.add_argument("--metadata-path", required=True, help="Path to metadata JSON file.")
    parser.add_argument("--output-path", required=True, help="Path to output predictions CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictor = HeartRiskPredictor(config=AppConfig())
    predictor.load_model(args.model_path, args.metadata_path)
    result = predictor.predict_from_csv(args.test_path, args.output_path)
    print(f"Predictions saved: {result['output_path']}")
    print(f"Rows predicted: {result['rows']}")


if __name__ == "__main__":
    main()
