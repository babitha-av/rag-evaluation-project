import csv
from pathlib import Path

from openai import AsyncOpenAI, OpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision,
)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "evaluation" / "results" / "rag_outputs.csv"
OUTPUT_PATH = BASE_DIR / "evaluation" / "results" / "ragas_results.csv"


def create_evaluator_llm():
    """
    Create a Ragas evaluator using the local Ollama model.
    """

    client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    return llm_factory(
        "qwen2.5:7b",
        provider="openai",
        client=client,
    )


def create_evaluator_embeddings():
    client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    return OpenAIEmbeddings(
        model="nomic-embed-text",
        client=client,
    )


def load_evaluation_data():
    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def run_evaluation():

    dataset = load_evaluation_data()

    evaluator_llm = create_evaluator_llm()
    evaluator_embeddings = create_evaluator_embeddings()

    faithfulness = Faithfulness(llm=evaluator_llm)

    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    answer_correctness = AnswerCorrectness(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    context_precision = ContextPrecision(llm=evaluator_llm)
    results = []

    # Only evaluate cases suitable for standard RAG metrics.
    excluded_categories = {
        "negative",
        "ambiguous",
        "adversarial",
    }

    evaluation_cases = [
        row
        for row in dataset
        if row["category"] not in excluded_categories
        and row["reference_context"].strip()
    ]

    print(
        f"Evaluating {len(evaluation_cases)} RAG quality cases "
        f"out of {len(dataset)} total cases.\n"
    )

    for index, row in enumerate(evaluation_cases, start=1):

        print(
            f"[{index}/{len(evaluation_cases)}] "
            f"{row['id']} - {row['question']}"
        )

        retrieved_contexts = [
            context.strip()
            for context in row["retrieved_context"].split("\n\n---\n\n")
            if context.strip()
        ]

        # Faithfulness
        faithfulness_result = faithfulness.score(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=retrieved_contexts,
        )

        # Answer Relevancy
        answer_relevancy_result = answer_relevancy.score(
            user_input=row["question"],
            response=row["answer"],
        )

        # Context Precision
        context_precision_result = context_precision.score(
            user_input=row["question"],
            reference=row["ground_truth"],
            retrieved_contexts=retrieved_contexts,
        )
        answer_correctness_result = answer_correctness.score(
            user_input=row["question"],
            response=row["answer"],
            reference=row["ground_truth"],
        )

        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "answer": row["answer"],
                "faithfulness": faithfulness_result.value,
                "answer_relevancy": answer_relevancy_result.value,
                "context_precision": context_precision_result.value,
                "answer_correctness": answer_correctness_result.value,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question",
        "category",
        "difficulty",
        "answer",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nRagas evaluation completed.")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()
