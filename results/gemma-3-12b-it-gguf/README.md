# gemma-3-12b-it-gguf

- Repo: `unsloth/gemma-3-12b-it-GGUF`
- File: `gemma-3-12b-it-Q4_K_M.gguf`
- Dataset: `~/Desktop/datasets/full-textgen-evalset`
- Launch script: `~/Desktop/llama-bash/gemma3-12b.sh`
- Endpoint: `http://127.0.0.1:8095/v1`
- Eval: n=100 per benchmark, timeout=180s, max_tokens=2048

## Accuracy

| benchmark | accuracy | correct/total | elapsed_sec | source |
|---|---:|---:|---:|---|
| mmlu | 33.0% | 33/100 | 96.82 | `results/gemma-3-12b-it-gguf/text/results/full_dataset_20260604_155307/fast_textgen_eval_20260604_075307_mmlu.json` |
| geo-mmlu-high-school | 81.0% | 81/100 | 85.3 | `results/gemma-3-12b-it-gguf/text/results/full_dataset_20260604_155307/fast_textgen_eval_20260604_075307_geo-mmlu-high-school.json` |
| law-mmlu-professional | 38.0% | 38/100 | 256.52 | `results/gemma-3-12b-it-gguf/text/results/full_dataset_20260604_155307/fast_textgen_eval_20260604_075307_law-mmlu-professional.json` |
| gsm8k | 79.0% | 79/100 | 1472.49 | `results/gemma-3-12b-it-gguf/text/results/full_dataset_20260604_155307/fast_textgen_eval_20260604_075307_gsm8k.json` |
| humaneval | 89.0% | 89/100 | 1638.3 | `results/gemma-3-12b-it-gguf/text/results/full_dataset_20260604_155307/fast_textgen_eval_20260604_075307_humaneval.json` |

## Speed sample

```text
prompt eval time =     458.95 ms /    13 tokens (   35.30 ms per token,    28.33 tokens per second)
       eval time =      77.35 ms /     2 tokens (   38.68 ms per token,    25.86 tokens per second)
      total time =     536.31 ms /    15 tokens
prompt eval time =    1307.85 ms /   147 tokens (    8.90 ms per token,   112.40 tokens per second)
       eval time =   13742.48 ms /   180 tokens (   76.35 ms per token,    13.10 tokens per second)
      total time =   15050.33 ms /   327 tokens
```

Updated: 2026-06-04T16:52:17.418248+08:00
