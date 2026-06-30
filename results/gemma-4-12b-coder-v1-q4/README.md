# gemma-4-12b-coder-v1-q4

## Summary
- Hugging Face repo: `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF`
- GGUF: `/Users/yenhaohuang/Desktop/models/gemma-4-12b-coder-fable5-composer2.5-v1-gguf/gemma4-coding-Q4_K_M.gguf`
- Original launch script: `/Users/yenhaohuang/Desktop/llama-bash/gemma-4-12b-coder-v1-q4.sh`
- Corrected no-think launch script: `/Users/yenhaohuang/Desktop/llama-bash/gemma-4-12b-coder-v1-q4-no-think.sh`
- Server: `llama-server` on port `8098`, `--ctx-size 8192`, `--temp 0.0`, `--top-p 0.95`
- Dataset root for mmlu/gsm8k/humaneval: `/Users/yenhaohuang/Desktop/datasets/full-textgen-evalset`
- TruthfulQA source: `/Users/yenhaohuang/Desktop/model-tester/deprecated/gpt-oss-20b/eval/datasets/truthfulqa.jsonl`
- Per-request eval timeout: 180s

## Accuracy (100 questions per benchmark)

| Benchmark | Accuracy | Correct/Total | Source JSON | Notes |
|---|---:|---:|---|---|
| mmlu | 83.0% | 83/100 | `raw/fast_textgen_eval_20260626_045232_mmlu.json` | original run |
| gsm8k | 98.0% | 98/100 | `raw/fast_textgen_eval_20260626_054103_gsm8k.json` | original run |
| truthfulqa | 76.0% | 76/100 | `../gemma-4-12b-coder-v1-q4-no-think/raw/truthfulqa_mcq_20260630_065647_truthfulqa.json` | corrected no-think run |
| humaneval | 87.0% | 87/100 | `../gemma-4-12b-coder-v1-q4-no-think/raw/fast_textgen_eval_20260630_065847_humaneval.json` | corrected no-think run |

## Important correction

The first TruthfulQA/HumanEval run accidentally used Gemma thinking mode (`thinking = 1` in llama-server). That produced invalid low scores:

| Benchmark | Thinking-enabled run | Correct no-think run |
|---|---:|---:|
| truthfulqa | 3.0% | 76.0% |
| humaneval | 11.0% | 87.0% |

Use the no-think scores above for model comparison. The corrected run used:

```bash
--reasoning off --chat-template-kwargs '{"enable_thinking": false}'
```

and the resulting JSON files have `thinking` empty for all 100/100 questions.

## Speed sample

```text
prompt eval time =    1118.06 ms /   115 tokens (    9.72 ms per token,   102.86 tokens per second)
       eval time =   14023.38 ms /   180 tokens (   77.91 ms per token,    12.84 tokens per second)
      total time =   15141.43 ms /   295 tokens
```

## Files
- Original raw responses: `raw/`
- Corrected no-think raw responses: `../gemma-4-12b-coder-v1-q4-no-think/raw/`
- Logs: `logs/`
- Source index: `text_sources.json`
- Corrected no-think source index: `../gemma-4-12b-coder-v1-q4-no-think/text_sources.json`
- Job status: `job_status.json`
- File tree guard output: `file_tree_guard.txt`
