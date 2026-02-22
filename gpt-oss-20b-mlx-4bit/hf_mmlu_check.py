import requests
from datetime import datetime

MODELS = [
    "InferenceIllusionist/gpt-oss-20b-MLX-4bit",
    "openai/gpt-oss-20b",
]

API = "https://huggingface.co/api/models/{}"

print(f"timestamp={datetime.now().isoformat()}")
for model_id in MODELS:
    print(f"\n=== {model_id} ===")
    r = requests.get(API.format(model_id), timeout=30)
    print(f"status={r.status_code}")
    if r.status_code != 200:
        print(r.text[:300])
        continue

    data = r.json()
    card = data.get("cardData", {}) or {}
    model_index = card.get("model-index", []) or []

    found = []
    for entry in model_index:
        for result in entry.get("results", []) or []:
            task = (result.get("task") or {}).get("name", "")
            dataset = (result.get("dataset") or {}).get("name", "")
            for metric in result.get("metrics", []) or []:
                name = metric.get("name", "")
                value = metric.get("value", None)
                if any(k in (task + dataset + name).lower() for k in ["mmlu", "massive multitask language understanding"]):
                    found.append({
                        "task": task,
                        "dataset": dataset,
                        "metric": name,
                        "value": value,
                    })

    if not found:
        print("No MMLU metrics found in cardData.model-index.")
    else:
        for row in found:
            print(f"task={row['task']} | dataset={row['dataset']} | metric={row['metric']} | value={row['value']}")
