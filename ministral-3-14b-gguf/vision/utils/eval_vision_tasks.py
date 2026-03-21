#!/usr/bin/env python3
import io
import re
import json
import base64
import time
import argparse
from pathlib import Path

import requests
from datasets import load_from_disk


def to_data_url(pil_img):
    # Some datasets contain CMYK images; convert before PNG encode.
    if pil_img.mode not in ("RGB", "RGBA", "L"):
        pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def ask_with_image(api, model, text, image_url=None, pil_img=None, max_tokens=120, retries=3):
    content = [{"type": "text", "text": text}]
    if image_url is not None:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    elif pil_img is not None:
        content.append({"type": "image_url", "image_url": {"url": to_data_url(pil_img)}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    last_err = None
    for _ in range(max(1, retries)):
        try:
            r = requests.post(api, json=payload, timeout=180)
            if r.status_code != 200:
                last_err = f"[ERROR] HTTP {r.status_code}: {r.text[:600]}"
                continue
            j = r.json()
            txt = j["choices"][0]["message"]["content"]
            return txt, txt
        except Exception as e:
            last_err = f"[ERROR] {e}"
            time.sleep(0.8)
    return None, last_err or "[ERROR] unknown"


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def run_ocr(ds_path, out_jsonl, api, model, n=100):
    ds = load_from_disk(ds_path)
    total = min(n, len(ds))
    passed = 0
    ensure_parent(Path(out_jsonl))
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(total):
            row = ds[i]
            query = 'OCR task: read this Pokemon card and answer strictly JSON with keys "name" and "hp" (hp as integer).'
            pred, resp = ask_with_image(api, model, query, image_url=row["image_url"], max_tokens=80)

            ok = False
            gold_name = str(row.get("name", "")).strip().lower()
            gold_hp = str(row.get("hp", "")).strip()
            if pred:
                low = pred.lower()
                if gold_name and (gold_name in low) and gold_hp and (gold_hp in low):
                    ok = True
            if ok:
                passed += 1

            rec = {
                "idx": i,
                "query": query,
                "gold": {"name": row.get("name"), "hp": row.get("hp")},
                "pred": pred,
                "passed": ok,
                "skipped": False,
                "response": resp,
                "image_url": row.get("image_url"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"OCR {i+1}/{total} pass={passed}", flush=True)

    return {"total": total, "passed": passed, "accuracy": passed / max(1, total)}


def run_classification(ds_path, out_jsonl, api, model, n=100):
    ds = load_from_disk(ds_path)
    labels = ds.features["label"].names
    total = min(n, len(ds))
    passed = 0
    ensure_parent(Path(out_jsonl))

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(total):
            row = ds[i]
            gold = labels[row["label"]]
            query = (
                "Image classification task: choose one label from "
                "[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]. "
                "Output one label only."
            )
            pred, resp = ask_with_image(api, model, query, pil_img=row["img"], max_tokens=20)
            norm = re.sub(r"[^a-z]", "", (pred or "").strip().lower())
            gnorm = re.sub(r"[^a-z]", "", gold.lower())
            ok = norm == gnorm
            if ok:
                passed += 1

            rec = {
                "idx": i,
                "query": query,
                "gold": gold,
                "pred": pred,
                "passed": ok,
                "skipped": False,
                "response": resp,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"CLS {i+1}/{total} pass={passed}", flush=True)

    return {"total": total, "passed": passed, "accuracy": passed / max(1, total)}


def run_detection(ds_path, out_jsonl, api, model, n=100):
    ds = load_from_disk(ds_path)
    cat_names = ds.features["objects"]["category"].feature.names
    valid = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]
    total = min(n, len(ds))
    passed = 0
    ensure_parent(Path(out_jsonl))

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(total):
            row = ds[i]
            query = (
                "Object detection task: list which PPE classes are present in the image from "
                "[Coverall, Face_Shield, Gloves, Goggles, Mask]. "
                "Return comma-separated class names only."
            )
            pred, resp = ask_with_image(api, model, query, pil_img=row["image"], max_tokens=40)

            gold_set = {cat_names[c] for c in row["objects"]["category"]}
            pred_set = set()
            if pred:
                lower = pred.lower()
                for k in valid:
                    if k.lower() in lower:
                        pred_set.add(k)
            ok = pred_set == gold_set
            if ok:
                passed += 1

            rec = {
                "idx": i,
                "query": query,
                "gold": sorted(gold_set),
                "pred": pred,
                "passed": ok,
                "skipped": False,
                "response": resp,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"DET {i+1}/{total} pass={passed}", flush=True)

    return {"total": total, "passed": passed, "accuracy": passed / max(1, total)}


def main():
    ap = argparse.ArgumentParser(description="Evaluate OCR / classification / detection image tasks via OpenAI-compatible llama-server endpoint")
    ap.add_argument("--api", default="http://127.0.0.1:3172/v1/chat/completions")
    ap.add_argument("--model", default="ministral-3-14b")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out-dir", default="/Users/yenhaohuang/Desktop/model-tester/ministral-3-14b-vision/results")
    ap.add_argument("--ocr-ds", default="/Users/yenhaohuang/Desktop/datasets/vision-eval-gpt-oss-20b/ocr_TheFusion21_PokemonCards_train_300")
    ap.add_argument("--cls-ds", default="/Users/yenhaohuang/Desktop/datasets/vision-eval-gpt-oss-20b/cls_cifar10_train_1000")
    ap.add_argument("--det-ds", default="/Users/yenhaohuang/Desktop/datasets/vision-eval-gpt-oss-20b/det_cppe5_train_500")
    ap.add_argument("--tasks", default="ocr,classification,detection", help="comma-separated subset: ocr,classification,detection")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {x.strip() for x in args.tasks.split(",") if x.strip()}

    t0 = time.time()
    ocr = run_ocr(args.ocr_ds, str(out_dir / f"ocr_pokemoncards_{args.num}.jsonl"), args.api, args.model, n=args.num) if "ocr" in wanted else None
    cls = run_classification(args.cls_ds, str(out_dir / f"classification_cifar10_{args.num}.jsonl"), args.api, args.model, n=args.num) if "classification" in wanted else None
    det = run_detection(args.det_ds, str(out_dir / f"detection_cppe5_{args.num}.jsonl"), args.api, args.model, n=args.num) if "detection" in wanted else None

    summary = {
        "model": args.model,
        "api": args.api,
        "num_per_task": args.num,
        "tasks": sorted(list(wanted)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "ocr": ocr,
        "classification": cls,
        "detection": det,
    }
    (out_dir / f"summary_{args.num}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
