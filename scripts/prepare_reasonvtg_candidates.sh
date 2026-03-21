#!/usr/bin/env bash

set -euo pipefail

# CPU-side data preparation for ReasonVTG-Bench candidates.
#
# Example:
#   bash scripts/prepare_reasonvtg_candidates.sh \
#     --output_jsonl output/reasonvtg_candidates_all.jsonl \
#     --max_per_video 2 \
#     --keep_existing_reasoning

output_jsonl="output/reasonvtg_candidates_all.jsonl"
max_per_video=2
keep_existing_reasoning=false
seed=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_jsonl) output_jsonl="$2"; shift 2 ;;
    --max_per_video) max_per_video="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --keep_existing_reasoning) keep_existing_reasoning=true; shift 1 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

bench_files=(
  data/TimeLens-Bench/activitynet-timelens.json
  data/TimeLens-Bench/charades-timelens.json
  data/TimeLens-Bench/qvhighlights-timelens.json
)

extra=()
if [[ "${keep_existing_reasoning}" == "true" ]]; then
  extra+=(--keep_existing_reasoning)
fi

python scripts/build_reasonvtg_bench.py \
  --bench_json "${bench_files[@]}" \
  --output_jsonl "${output_jsonl}" \
  --max_per_video "${max_per_video}" \
  --seed "${seed}" \
  "${extra[@]}"

