#!/usr/bin/env bash

set -euo pipefail

# CPU-side data preparation for EG-CoT SFT.
#
# Example:
#   bash scripts/prepare_egcot_data.sh \
#     --input_jsonl data/TimeLens-100K/timelens-100k.jsonl \
#     --output_jsonl output/TimeLens-8B/sft/egcot_timelens100k.jsonl \
#     --target_reasoning_ratio 0.4

input_jsonl=""
output_jsonl=""
target_reasoning_ratio=0.4
llm_provider="none"
llm_model="gpt-4o-mini"
llm_cache_jsonl="output/egcot_llm_cache.jsonl"
max_samples=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_jsonl) input_jsonl="$2"; shift 2 ;;
    --output_jsonl) output_jsonl="$2"; shift 2 ;;
    --target_reasoning_ratio) target_reasoning_ratio="$2"; shift 2 ;;
    --llm_provider) llm_provider="$2"; shift 2 ;;
    --llm_model) llm_model="$2"; shift 2 ;;
    --llm_cache_jsonl) llm_cache_jsonl="$2"; shift 2 ;;
    --max_samples) max_samples="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${input_jsonl}" || -z "${output_jsonl}" ]]; then
  echo "Usage: bash scripts/prepare_egcot_data.sh --input_jsonl <path> --output_jsonl <path> [--target_reasoning_ratio 0.4]" >&2
  exit 1
fi

extra=()
if [[ -n "${max_samples}" ]]; then
  extra+=(--max_samples "${max_samples}")
fi

python scripts/build_egcot_data.py \
  --input_jsonl "${input_jsonl}" \
  --output_jsonl "${output_jsonl}" \
  --target_reasoning_ratio "${target_reasoning_ratio}" \
  --llm_provider "${llm_provider}" \
  --llm_model "${llm_model}" \
  --llm_cache_jsonl "${llm_cache_jsonl}" \
  "${extra[@]}"

