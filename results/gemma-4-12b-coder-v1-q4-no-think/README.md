# gemma-4-12b-coder-v1-q4-no-think

No-think rerun for the two suspicious low-score benchmarks from `gemma-4-12b-coder-v1-q4`.

## Launch
- Model: `/Users/yenhaohuang/Desktop/models/gemma-4-12b-coder-fable5-composer2.5-v1-gguf/gemma4-coding-Q4_K_M.gguf`
- Launch script: `/Users/yenhaohuang/Desktop/llama-bash/gemma-4-12b-coder-v1-q4-no-think.sh`
- Server: `llama-server` on port `8098`, `--ctx-size 8192`, `--temp 0.0`, `--top-p 0.95`
- No-think flags: `--reasoning off --chat-template-kwargs '{"enable_thinking": false}'`
- Ready gate: `/v1/chat/completions` smoke request, HTTP 200 with `choices` required before benchmark.

## Accuracy (100 questions per benchmark)

| Benchmark | Accuracy | Correct/Total | Source JSON | Thinking fields |
|---|---:|---:|---|---:|
| truthfulqa | 76.0% | 76/100 | `raw/truthfulqa_mcq_20260630_065647_truthfulqa.json` | 0/100 |
| humaneval | 87.0% | 87/100 | `raw/fast_textgen_eval_20260630_065847_humaneval.json` | 0/100 |

## Comparison against the earlier thinking-enabled run

| Benchmark | Earlier run | No-think rerun | Delta |
|---|---:|---:|---:|
| truthfulqa | 3.0% | 76.0% | +73.0 pp |
| humaneval | 11.0% | 87.0% | +76.0 pp |

The earlier low scores were caused by running with Gemma thinking enabled (`thinking = 1` in llama-server). TruthfulQA put the useful answer in `reasoning_content` while `content` was empty, and HumanEval often emitted explanation/markdown instead of clean executable Python. This no-think run is the valid score for those two benchmarks.

## Files
- Raw responses: `raw/`
- Logs: `logs/`
- Source index: `text_sources.json`
- Job status: `job_status.json`
