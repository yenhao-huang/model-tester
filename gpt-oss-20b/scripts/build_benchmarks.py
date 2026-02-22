#!/usr/bin/env python3
"""Convert parquet benchmark datasets to JSONL for the gpt-oss-20b runner."""

import json
import random
from pathlib import Path

import duckdb

DATASET_BASE = Path.home() / "Desktop" / "datasets" / "eval_text_gen"
OUT_BASE = Path(__file__).resolve().parent.parent / "eval" / "datasets"


def build_mmlu():
    """MMLU: multiple-choice knowledge/reasoning."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT question, choices, answer, subject
        FROM '{DATASET_BASE}/reasoning/mmlu_test.parquet'
    """).fetchall()
    con.close()

    labels = ["A", "B", "C", "D"]
    cases = []
    for i, (question, choices, answer_idx, subject) in enumerate(rows):
        options = "\n".join(f"{labels[j]}. {c}" for j, c in enumerate(choices))
        prompt = f"Answer the following multiple-choice question. Reply with ONLY the letter (A, B, C, or D).\n\nSubject: {subject}\nQuestion: {question}\n{options}"
        cases.append({
            "id": f"mmlu_{i}",
            "category": "reasoning",
            "prompt": prompt,
            "expected": labels[answer_idx],
            "checks": ["non_empty", "mcq_letter"],
            "meta": {"subject": subject, "correct": labels[answer_idx]},
        })
    return cases


def build_hellaswag():
    """HellaSwag: commonsense sentence completion."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT activity_label, ctx, endings, label
        FROM '{DATASET_BASE}/reasoning/hellaswag_validation.parquet'
    """).fetchall()
    con.close()

    labels = ["A", "B", "C", "D"]
    cases = []
    for i, (activity, ctx, endings, label_idx) in enumerate(rows):
        label_int = int(label_idx) if isinstance(label_idx, str) else label_idx
        options = "\n".join(f"{labels[j]}. {e}" for j, e in enumerate(endings))
        prompt = f"Pick the most plausible continuation. Reply with ONLY the letter (A, B, C, or D).\n\nContext ({activity}): {ctx}\n{options}"
        cases.append({
            "id": f"hellaswag_{i}",
            "category": "reasoning",
            "prompt": prompt,
            "expected": labels[label_int],
            "checks": ["non_empty", "mcq_letter"],
            "meta": {"correct": labels[label_int]},
        })
    return cases


def build_ifeval():
    """IFEval: instruction following."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT prompt, instruction_id_list, kwargs
        FROM '{DATASET_BASE}/instruction-following/ifeval.parquet'
    """).fetchall()
    con.close()

    cases = []
    for i, (prompt, instr_ids, kwargs) in enumerate(rows):
        cases.append({
            "id": f"ifeval_{i}",
            "category": "instruction-following",
            "prompt": prompt,
            "expected": None,
            "checks": ["non_empty", "ifeval_instructions"],
            "meta": {"instruction_ids": instr_ids, "kwargs": kwargs},
        })
    return cases


def build_humaneval():
    """HumanEval: code generation."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT task_id, prompt, canonical_solution, test, entry_point
        FROM '{DATASET_BASE}/code-generation/humaneval.parquet'
    """).fetchall()
    con.close()

    cases = []
    for i, (task_id, prompt_code, solution, test, entry_point) in enumerate(rows):
        user_prompt = f"Complete the following Python function. Return ONLY the function body (no explanation, no markdown).\n\n{prompt_code}"
        cases.append({
            "id": f"humaneval_{i}",
            "category": "code-generation",
            "prompt": user_prompt,
            "expected": solution,
            "checks": ["non_empty", "code_executes"],
            "meta": {"task_id": task_id, "test": test, "entry_point": entry_point, "prompt_code": prompt_code},
        })
    return cases


def build_truthfulqa():
    """TruthfulQA: factual correctness (multiple choice)."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT question, mc1_targets
        FROM '{DATASET_BASE}/truthfulness/truthfulqa_mc.parquet'
    """).fetchall()
    con.close()

    labels = list("ABCDEFGHIJKLMNOP")
    cases = []
    for i, (question, mc1) in enumerate(rows):
        if isinstance(mc1, str):
            mc1 = json.loads(mc1)
        choices = mc1.get("choices", [])
        label_list = mc1.get("labels", [])
        if not choices:
            continue
        correct_idx = label_list.index(1) if 1 in label_list else 0
        options = "\n".join(f"{labels[j]}. {c}" for j, c in enumerate(choices[:len(labels)]))
        prompt = f"Answer the following question truthfully. Reply with ONLY the letter.\n\nQuestion: {question}\n{options}"
        cases.append({
            "id": f"truthfulqa_{i}",
            "category": "truthfulness",
            "prompt": prompt,
            "expected": labels[correct_idx],
            "checks": ["non_empty", "mcq_letter"],
            "meta": {"correct": labels[correct_idx]},
        })
    return cases


def write_jsonl(cases, name):
    out = OUT_BASE / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✅ {name}: {len(cases)} cases → {out}")


def main():
    print("Building benchmark JSONL files...\n")
    for name, builder in [
        ("mmlu", build_mmlu),
        ("hellaswag", build_hellaswag),
        ("ifeval", build_ifeval),
        ("humaneval", build_humaneval),
        ("truthfulqa", build_truthfulqa),
    ]:
        try:
            cases = builder()
            write_jsonl(cases, name)
        except Exception as e:
            print(f"❌ {name}: {e}")
    print("\nDone!")


if __name__ == "__main__":
    main()
