import json, random

with open("results/qwen2.5_7b/outputs.json") as f:
    records = json.load(f)


followers = [r for r in records
             if r["perturbation_type"] == "entity_swap"
             and not r["generation"]["abstained"]
             and not r["scores"]["correctness_em"]
             and r["scores"]["faithfulness_score"] > 0.7]

for r in random.sample(followers, min(3, len(followers))):
    print("Q:", r["question"])
    print("Context:", r["perturbed_context"])
    print("Answer:", r["generation"]["answer"])
    print("Faithfulness:", r["scores"]["faithfulness_score"])
    print()