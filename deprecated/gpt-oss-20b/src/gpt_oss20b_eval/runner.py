import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from .client import build_client, run_inference
from .guardrails import run_checks
from .schemas import EvalResult, TestCase


def load_dataset(path: Path) -> list[TestCase]:
    rows: list[TestCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(TestCase.model_validate_json(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="gpt-oss-20b eval runner")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    load_dotenv()
    dataset_path = Path(args.dataset)
    prompt_path = Path(args.prompt)
    out_path = Path(args.out)

    system_prompt = prompt_path.read_text(encoding="utf-8")
    cases = load_dataset(dataset_path)

    client = build_client()
    results: list[EvalResult] = []

    for tc in cases:
        t0 = time.time()
        response = run_inference(client, system_prompt, tc.prompt)
        latency_ms = int((time.time() - t0) * 1000)
        check_map = run_checks(response, tc.checks)
        passed = all(check_map.values())

        results.append(
            EvalResult(
                id=tc.id,
                category=tc.category,
                prompt=tc.prompt,
                response=response,
                checks=check_map,
                passed=passed,
                latency_ms=latency_ms,
            )
        )

    total = len(results)
    passed_n = sum(1 for r in results if r.passed)
    report = {
        "summary": {
            "total": total,
            "passed": passed_n,
            "pass_rate": (passed_n / total) if total else 0,
            "avg_latency_ms": (sum(r.latency_ms for r in results) / total) if total else 0,
        },
        "results": [r.model_dump() for r in results],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
