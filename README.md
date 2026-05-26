# FinMTEB SOTA Reranker

This project targets the FinanceMTEB Chinese reranking slice:

- `FinanceMTEB/FinEvaRetrieval-reranking`
- `FinanceMTEB/DISCFinLLM-reranking`

The achieved SOTA path is:

1. Reproduce the official leaderboard snapshot.
2. Run a zero-shot Qwen3 reranker with train-only score-mode and rank-fusion selection.
3. Freeze the selected strategy, then evaluate on untouched test splits.

On the L20 box, `Qwen/Qwen3-Reranker-8B` with `true_logit` scoring and per-query
RRF lexical fusion reached `0.997807` MAP average on `Reranking_zh`, above the
official `benchmark.xlsx` snapshot top average of `0.993100`.

See `RESULTS.md` for exact scores, commands, and environment details. The
public snapshot comparison is recorded in
`reports/public_reranking_zh_snapshot_comparison.md`.

## Setup

Use Python 3.10, 3.11, or 3.12 on the GPU machine.

```bash
cd /Users/yinxiaogou/Documents/resume/finmteb-sota-reranker
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[train]"
```

Login to Hugging Face if you plan to push the adapter:

```bash
huggingface-cli login
```

## Leaderboard Snapshot

```bash
python scripts/finmteb_leaderboard_snapshot.py
```

Current target from the official `benchmark.xlsx` snapshot:

- `Reranking_zh` top average is about `0.9931`.
- `FinEvaReranking` is saturated at `1.0000`.
- `DISCFinLLMReranking` top visible score is about `0.9956`.

The current result beats the official visible Chinese reranking average, with
`DISCFinLLMReranking` test MAP `0.995614`.

## Current Best Zero-Shot Run

Search score mode and rank-fusion blends on train only:

```bash
python scripts/search_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --score-mode true_logit \
  --cache-tag qwen3_8b_true_logit \
  --keep-candidates 25 \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1.json
```

Audit the frozen choice on train-only CV:

```bash
python scripts/validate_blend_strategy_cv.py \
  --tasks zh \
  --split train \
  --cache-tag qwen3_8b_true_logit \
  --folds 7 \
  --seed 20260526 \
  --keep-candidates 20 \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_nested_cv_v1.json
```

Then freeze the train-selected strategy for test:

```bash
python scripts/eval_blend_strategy.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split test \
  --strategy-file reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1.json \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --score-mode true_logit \
  --cache-tag qwen3_8b_true_logit \
  --output reports/qwen3_reranker_8b_zh_true_logit_blend_strategy_v1_test.json
```

## Previous Baseline

The earlier instruction/alpha search is still useful as a simple reproduction
baseline.

```bash
python scripts/search_instruction_alpha.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --output reports/qwen3_reranker_8b_train_search.json
```

Then freeze the selected instruction and alpha for test:

```bash
python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split test \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --tuning-file reports/qwen3_reranker_8b_train_search.json \
  --output reports/qwen3_reranker_8b_test.json
```

If you want only alpha tuning with a fixed instruction:

```bash
python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --lexical-grid \
  --output reports/qwen3_reranker_8b_train_tune.json
```

If memory is tight, use `Qwen/Qwen3-Reranker-4B` first.

## QLoRA Training

```bash
accelerate launch scripts/train_qwen3_reranker_lora.py \
  --config configs/l20_qwen3_reranker_8b.yaml
```

Evaluate the adapter:

```bash
python scripts/search_instruction_alpha.py \
  --model Qwen/Qwen3-Reranker-8B \
  --adapter outputs/qwen3-reranker-8b-finmteb-lora \
  --tasks zh \
  --split train \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --output reports/qwen3_8b_lora_train_search.json
```

```bash
python scripts/eval_qwen3_reranker.py \
  --model Qwen/Qwen3-Reranker-8B \
  --adapter outputs/qwen3-reranker-8b-finmteb-lora \
  --tasks zh \
  --split test \
  --batch-size 8 \
  --max-length 4096 \
  --load-in-4bit \
  --tuning-file reports/qwen3_8b_lora_train_search.json \
  --output reports/qwen3_8b_lora_test.json
```

## Guardrails

- Do not tune on test splits.
- Tune the lexical blend on train or validation only, then freeze it.
- Report per-task MAP/MRR/nDCG, not just the average.
- Save every command and commit hash before publishing.
