import json

with open("eval_results_fireworks.json", "r", encoding="utf-8") as f:
    fireworks = json.load(f)

with open("eval_results_openai.json", "r", encoding="utf-8") as f:
    openai_results = json.load(f)

for i, (fw, oa) in enumerate(zip(fireworks, openai_results), start=1):
    print(f"\n{'='*80}")
    print(f"QUESTION {i}: {fw['question']}")
    print(f"{'='*80}")

    print("\nGROUND TRUTH:")
    print(fw["ground_truth"])

    print("\nFIREWORKS ANSWER:")
    print(fw["answer"][:1000])

    print("\nOPENAI ANSWER:")
    print(oa["answer"][:1000])