import json
import os
import time
from eval_dataset import eval_rows
from app.graphs.simple_agent import graph
from langchain_core.messages import HumanMessage


def run_rag(question):
    provider = os.environ.get("MODEL_PROVIDER", "fireworks")
    model_name = (
        os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        if provider == "openai"
        else os.environ.get("FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b")
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=question + " Use retrieve_information tool.")
            ]
        },
        config={
            "tags": [provider, "activity-1", "rag-eval"],
            "metadata": {
                "provider": provider,
                "model_name": model_name,
                "evaluation_type": "activity_1_rag_eval",
            },
            "run_name": f"rag_eval_{provider}",
        },
    )

    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content:
            return msg.content

    return ""


def evaluate():
    results = []
    provider = os.environ.get("MODEL_PROVIDER", "fireworks")

    for row in eval_rows:
        question = row["question"]
        ground_truth = row["ground_truth"]

        answer = run_rag(question)
        time.sleep(10)

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "provider": provider,
            }
        )

        print("\n---")
        print("PROVIDER:", provider)
        print("QUESTION:", question)
        print("ANSWER:", answer[:300])

    return results


if __name__ == "__main__":
    results = evaluate()

    provider = os.environ.get("MODEL_PROVIDER", "fireworks")
    output_file = f"eval_results_{provider}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_file}")