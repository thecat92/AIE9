import json
import pandas as pd
from dotenv import load_dotenv
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall
from ragas.llms import llm_factory
from openai import OpenAI
import math

load_dotenv()

def load_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_ragas_rows(results):
    rows = []
    for row in results:
        rows.append(
            {
                "user_input": row["question"],
                "response": row["answer"],
                "reference": row["ground_truth"],
                "retrieved_contexts": [row["ground_truth"]],
            }
        )
    return rows

def evaluate_in_chunks(rows, label, chunk_size=10, model="gpt-4o-mini", max_tokens=512):
    """
    Evaluate rows (list of ragas-style rows) in chunks to avoid large single requests.
    Returns a DataFrame with concatenated results.
    """
    total = len(rows)
    chunks = math.ceil(total / chunk_size)
    all_dfs = []

    # create evaluator LLM once (shorter outputs + deterministic)
    evaluator_llm = llm_factory(
        model,
        client=OpenAI(),
        # These kwargs are forwarded to the underlying client; most LLM wrappers accept model_kwargs.
        model_kwargs={"max_tokens": max_tokens, "temperature": 0}
    )

    for i in range(chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        subset = rows[start:end]
        print(f"Evaluating chunk {i+1}/{chunks} — rows {start}..{end-1}")

        dataset = EvaluationDataset.from_list(subset)
        result = evaluate(
            dataset=dataset,
            metrics=[
                LLMContextRecall(),
                Faithfulness(),
                FactualCorrectness(),
            ],
            llm=evaluator_llm,
        )

        df = result.to_pandas()
        all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def run_eval(results_path: str, label: str, chunk_size=10, model="gpt-4o-mini", max_tokens=512):
    results = load_results(results_path)
    rows = to_ragas_rows(results)

    df = evaluate_in_chunks(rows, label, chunk_size=chunk_size, model=model, max_tokens=max_tokens)

    out_csv = f"ragas_scores_{label}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.mean(numeric_only=True))

if __name__ == "__main__":
    # tune chunk_size and max_tokens if you still hit truncation:
    # - smaller chunk_size -> fewer tokens per request
    # - smaller max_tokens -> shorter evaluator replies
    run_eval("eval_results_fireworks.json", "fireworks", chunk_size=8, model="gpt-4o-mini", max_tokens=512)
    run_eval("eval_results_openai.json", "openai", chunk_size=8, model="gpt-4o-mini", max_tokens=512)