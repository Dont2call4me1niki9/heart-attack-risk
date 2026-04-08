import argparse

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send demo request to FastAPI service.")
    parser.add_argument("--csv-path", required=True, help="Path to input CSV.")
    parser.add_argument("--output-path", required=True, help="Path to save predictions CSV.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/predict",
        help="FastAPI predict endpoint URL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = {
        "csv_path": args.csv_path,
        "output_path": args.output_path,
    }

    response = requests.post(args.url, json=payload, timeout=60)
    print("Status code:", response.status_code)
    print("Response:", response.json())


if __name__ == "__main__":
    main()
