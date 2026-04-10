#!/usr/bin/env python3
"""OpenRouter API caller.

Input  : a question/prompt string
Output : OpenRouter chat-completion response (content, thinking, usage, latency)
         Result is also saved to results/openrouter/<timestamp>_<model>.json

API key is read from the project-root .env file (key: ``openrouter``).
It can be overridden with --api-key on the CLI or the ``api_key`` argument.

.env format (project root)
--------------------------
openrouter=sk-or-...

Usage
-----
# CLI (api-key optional when .env is present)
python utils/providers/call_openrouter_api.py \
    --model "anthropic/claude-3.5-sonnet" \
    --question "What is the capital of France?"

# As a module
from utils.providers.call_openrouter_api import call_openrouter

result = call_openrouter(
    question="What is the capital of France?",
    model="anthropic/claude-3.5-sonnet",
)
print(result["content"])
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logger = logging.getLogger(__name__)

# Project root = two levels up from this file (utils/providers/call_openrouter_api.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env_key(env_path: Path | None = None) -> str:
    """Read ``openrouter`` key from a .env file.

    Search order:
      1. ``env_path`` if explicitly provided
      2. .env in the current working directory
      3. .env in the project root
    """
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates += [Path.cwd() / ".env", _PROJECT_ROOT / ".env"]

    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                if k.strip().lower() == "openrouter":
                    val = v.strip().strip('"').strip("'")
                    if val:
                        logger.debug("Loaded API key from %s", path)
                        return val
    return ""


_DEFAULT_OUT_DIR = _PROJECT_ROOT / "results" / "openrouter"


def _save_result(record: dict, out_dir: Path) -> Path:
    """Write *record* to ``out_dir/<timestamp>_<model_slug>.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_slug = record.get("model", "unknown").replace("/", "_").replace(":", "_")
    path = out_dir / f"{ts}_{model_slug}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Result saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# Core caller
# ---------------------------------------------------------------------------

def call_openrouter(
    question: str,
    model: str,
    api_key: str = "",
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 4096,
    temperature: float = 0,
    timeout_sec: int = 120,
    site_url: str = "",
    site_name: str = "",
    out_dir: Path | str | None = None,
    reasoning_effort: str = "",
) -> dict:
    """Call the OpenRouter chat-completions endpoint and save the result.

    Parameters
    ----------
    question:     User prompt / question to send.
    model:        Model ID, e.g. ``"anthropic/claude-3.5-sonnet"``.
    api_key:      OpenRouter API key. If omitted, read from .env (``openrouter=``).
    system_prompt: System message prepended to the conversation.
    max_tokens:   Maximum tokens in the completion.
    temperature:  Sampling temperature (0 = deterministic).
    timeout_sec:  HTTP request timeout in seconds.
    site_url:     Optional – forwarded as ``HTTP-Referer`` header.
    site_name:    Optional – forwarded as ``X-Title`` header.
    out_dir:          Directory to save the JSON result.
                      Defaults to ``results/openrouter/`` under the project root.
                      Pass ``False`` or ``""`` to disable saving.
    reasoning_effort: Enable thinking. One of ``"low"``, ``"medium"``, ``"high"``.
                      Leave empty to disable (default).

    Returns
    -------
    dict with keys:
        timestamp_utc – ISO-8601 request timestamp (str)
        model         – model ID echoed from the response (str)
        system_prompt – system prompt sent (str)
        user_prompt   – user question sent (str)
        max_tokens    – max_tokens setting (int)
        temperature   – temperature setting (float)
        thinking      – chain-of-thought / reasoning_content if present (str)
        response      – assistant reply text (str)
        usage         – token usage dict or None
        elapsed_sec   – wall-clock latency (float)
        error         – error message string, or None on success
        saved_to      – path of the saved JSON file, or None
    """
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "") or _load_env_key()
    if not api_key:
        raise ValueError(
            "OpenRouter API key not found. "
            "Set 'openrouter=sk-or-...' in .env or pass --api-key."
        )

    timestamp_utc = datetime.now(timezone.utc).isoformat()

    # Resolve output directory (None → default, falsy str → disabled)
    if out_dir is None:
        _out = _DEFAULT_OUT_DIR
    elif out_dir:
        _out = Path(out_dir)
    else:
        _out = None  # saving disabled

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort in ("low", "medium", "high"):
        payload["reasoning"] = {"effort": reasoning_effort}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    req = urllib_request.Request(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    logger.info("→ OpenRouter  model=%s  question_len=%d", model, len(question))
    t0 = time.time()

    try:
        with urllib_request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        elapsed = round(time.time() - t0, 2)
        msg      = body["choices"][0]["message"]
        content  = (msg.get("content") or "").strip()
        # OpenRouter returns thinking in "reasoning"; raw Anthropic API uses "reasoning_content"
        thinking = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        usage    = body.get("usage")
        model_id = body.get("model", model)

        logger.info(
            "← OpenRouter  model=%s  elapsed=%.2fs  prompt_tokens=%s  completion_tokens=%s",
            model_id,
            elapsed,
            usage.get("prompt_tokens") if usage else "?",
            usage.get("completion_tokens") if usage else "?",
        )

        record = {
            "timestamp_utc": timestamp_utc,
            "model":         model_id,
            "system_prompt": system_prompt,
            "user_prompt":   question,
            "max_tokens":    max_tokens,
            "temperature":   temperature,
            "thinking":      thinking,
            "response":      content,
            "usage":         usage,
            "elapsed_sec":   elapsed,
            "error":         None,
            "saved_to":      None,
        }

    except HTTPError as exc:
        elapsed = round(time.time() - t0, 2)
        body_text = exc.read().decode("utf-8", errors="replace")
        err_msg = f"HTTP {exc.code}: {body_text}"
        logger.error("OpenRouter HTTP error: %s", err_msg)
        record = {
            "timestamp_utc": timestamp_utc, "model": model,
            "system_prompt": system_prompt, "user_prompt": question,
            "max_tokens": max_tokens, "temperature": temperature,
            "thinking": "", "response": "", "usage": None,
            "elapsed_sec": elapsed, "error": err_msg, "saved_to": None,
        }

    except URLError as exc:
        elapsed = round(time.time() - t0, 2)
        err_msg = f"URLError: {exc.reason}"
        logger.error("OpenRouter network error: %s", err_msg)
        record = {
            "timestamp_utc": timestamp_utc, "model": model,
            "system_prompt": system_prompt, "user_prompt": question,
            "max_tokens": max_tokens, "temperature": temperature,
            "thinking": "", "response": "", "usage": None,
            "elapsed_sec": elapsed, "error": err_msg, "saved_to": None,
        }

    except TimeoutError:
        elapsed = round(time.time() - t0, 2)
        err_msg = f"Request timed out after {timeout_sec}s"
        logger.error("OpenRouter timeout: %s", err_msg)
        record = {
            "timestamp_utc": timestamp_utc, "model": model,
            "system_prompt": system_prompt, "user_prompt": question,
            "max_tokens": max_tokens, "temperature": temperature,
            "thinking": "", "response": "", "usage": None,
            "elapsed_sec": elapsed, "error": err_msg, "saved_to": None,
        }

    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.time() - t0, 2)
        err_msg = f"{type(exc).__name__}: {exc}"
        logger.error("OpenRouter unexpected error: %s", err_msg)
        record = {
            "timestamp_utc": timestamp_utc, "model": model,
            "system_prompt": system_prompt, "user_prompt": question,
            "max_tokens": max_tokens, "temperature": temperature,
            "thinking": "", "response": "", "usage": None,
            "elapsed_sec": elapsed, "error": err_msg, "saved_to": None,
        }

    if _out is not None:
        saved_path = _save_result(record, _out)
        record["saved_to"] = str(saved_path)

    return record


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Call the OpenRouter API with a question.")
    p.add_argument("--api-key",     default="",
                   help="OpenRouter API key (sk-or-...). Defaults to 'openrouter' in .env")
    p.add_argument("--model",       required=True, help='Model ID, e.g. "anthropic/claude-3.5-sonnet"')
    p.add_argument("--question",    required=True, help="Question / prompt to send")
    p.add_argument("--system",      default="You are a helpful assistant.",
                   help="System prompt (default: 'You are a helpful assistant.')")
    p.add_argument("--max-tokens",  type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--timeout-sec", type=int, default=120)
    p.add_argument("--site-url",    default="", help="HTTP-Referer header value")
    p.add_argument("--site-name",   default="", help="X-Title header value")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["", "low", "medium", "high"],
                   help="Enable thinking (low/medium/high). Leave empty to disable.")
    p.add_argument("--out-dir",     default=None,
                   help="Directory to save JSON result (default: results/openrouter/). "
                        "Pass empty string to disable saving.")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    result = call_openrouter(
        question=args.question,
        model=args.model,
        api_key=args.api_key,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
        site_url=args.site_url,
        site_name=args.site_name,
        out_dir=args.out_dir,
        reasoning_effort=args.reasoning_effort,
    )

    if result["error"]:
        print(f"[ERROR] {result['error']}")
        return

    print("\n=== Response ===")
    if result["thinking"]:
        print(f"[thinking]\n{result['thinking']}\n")
    print(result["response"])
    print(f"\n[model={result['model']}  elapsed={result['elapsed_sec']}s  usage={result['usage']}]")
    if result["saved_to"]:
        print(f"[saved → {result['saved_to']}]")


if __name__ == "__main__":
    main()
