import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--image", required=True, help="Path to jpg/png image")
    args = parser.parse_args()

    api_url = args.api.rstrip("/")
    image_path = Path(args.image)

    print("Health:")
    print(requests.get(f"{api_url}/health", timeout=30).json())

    print("Classes:")
    print(requests.get(f"{api_url}/classes", timeout=30).json())

    with open(image_path, "rb") as img_file:
        response = requests.post(
            f"{api_url}/predict",
            files={"file": (image_path.name, img_file, "image/jpeg")},
            timeout=60,
        )
    response.raise_for_status()

    print("Prediction:")
    print(response.json())


if __name__ == "__main__":
    main()
