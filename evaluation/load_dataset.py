import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "evaluation_dataset.csv"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":

    dataset = load_dataset()

    print(f"Loaded {len(dataset)} test cases\n")

    for test_case in dataset[:5]:
        print(
            f"{test_case['id']} | "
            f"{test_case['category']} | "
            f"{test_case['question']}"
        )
