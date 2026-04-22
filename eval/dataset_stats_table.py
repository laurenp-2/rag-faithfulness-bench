import json, spacy
nlp = spacy.load("en_core_web_sm")

with open("data/base_qa_pairs.json") as f:
    pairs = json.load(f)

type_map = {"PERSON": "Person", "ORG": "Organization",
            "GPE": "Location", "LOC": "Location",
            "DATE": "Date/Number", "TIME": "Date/Number",
            "CARDINAL": "Date/Number", "ORDINAL": "Date/Number",
            "QUANTITY": "Date/Number"}

counts = {"Person": 0, "Organization": 0, "Location": 0, "Date/Number": 0, "Other": 0}
ctx_lens, q_lens = [], []

for p in pairs:
    doc = nlp(p["gold_answer"])
    label = "Other"
    if doc.ents:
        label = type_map.get(doc.ents[0].label_, "Other")
    counts[label] += 1
    ctx_lens.append(len(p["gold_context"].split()))
    q_lens.append(len(p["question"].split()))

n = len(pairs)
for k, v in counts.items():
    print(f"{k}: {v} ({v/n*100:.1f}%)")
print(f"Avg context tokens: {sum(ctx_lens)/n:.1f}")
print(f"Avg question tokens: {sum(q_lens)/n:.1f}")