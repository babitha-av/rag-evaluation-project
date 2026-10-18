import csv
from pathlib import Path

from app.rag_pipeline import RAGPipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation_dataset.csv"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
RESULTS_PATH = RESULTS_DIR / "rag_outputs.csv"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def run_evaluation():
    dataset = load_dataset()
    rag = RAGPipeline()

    results = []

    print(f"Running {len(dataset)} evaluation cases...\n")

    for index, test_case in enumerate(dataset, start=1):

        print(
            f"[{index}/{len(dataset)}] "
            f"{test_case['id']} - {test_case['question']}"
        )

        result = rag.ask(test_case["question"])

        retrieved_context = "\n\n---\n\n".join(result["context"])
        retrieved_sources = "\n".join(
            source for source in result["sources"] if source
        )

        results.append(
            {
                "id": test_case["id"],
                "question": test_case["question"],
                "ground_truth": test_case["ground_truth"],
                "expected_source": test_case["expected_source"],
                "reference_context": test_case["reference_context"],
                "category": test_case["category"],
                "difficulty": test_case["difficulty"],
                "expected_behavior": test_case["expected_behavior"],
                "answer": result["answer"],
                "retrieved_context": retrieved_context,
                "retrieved_sources": retrieved_sources,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question",
        "ground_truth",
        "expected_source",
        "reference_context",
        "category",
        "difficulty",
        "expected_behavior",
        "answer",
        "retrieved_context",
        "retrieved_sources",
    ]

    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nEvaluation completed.")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
