#!/usr/bin/env bash

# SFT Training Test Script for L40 GPUs (Single GPU smoke test)
# This is a minimal configuration for testing on a single L40
# before running full multi-GPU training

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1

# Model path - using local model directory
model_path="./model"
datasets="gemini_refined_data"
model_id="qwen3-vl-8b"

# ============ L40 Test Configuration (Minimal) ============
# Use very small token count for testing
min_tokens=64
total_tokens=3584        # Minimal for testing (~14GB VRAM)
fps=2
fps_max_frames=""

seed=42

# ============ Single GPU Test Configuration ============
global_batch_size=8      # Very small for testing
batch_per_device=1
num_devices=1              # Single GPU test
epochs=1
target_size=100            # Very small for quick test

# Use ZeRO-1 for single GPU (simpler, less memory overhead)
deepspeed_config="scripts/zero1.json"
output_root="output/TimeLens-8B/sft_test"
report_to="none"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --datasets) datasets="$2"; shift 2 ;;
    --min_tokens) min_tokens="$2"; shift 2 ;;
    --total_tokens) total_tokens="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --fps_max_frames) fps_max_frames="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --target_size) target_size="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate model path
if [[ ! -d "${model_path}" ]]; then
  echo "Error: Model path does not exist: ${model_path}"
  exit 1
fi

echo "========================================"
echo "L40 SFT TEST (Single GPU Smoke Test)"
echo "========================================"
echo "Model path: ${model_path}"
echo "Total tokens: ${total_tokens} (Test mode)"
echo "Target size: ${target_size} samples"
echo "Single GPU test mode"
echo "========================================"

# Calculate derived parameters
if [[ -z "${fps_max_frames}" ]]; then
  fps_max_frames=$((total_tokens / min_tokens * 2))
fi
run_tag="$(date +%Y%m%d-%H%M)"
run_name="sft-TEST-${run_tag}_MAXFRAMES-${fps_max_frames}_TOKENS-${total_tokens}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}"
echo "Output directory: ${output_dir}"

# Launch single GPU training
deepspeed --num_gpus=1 training/train/train_sft_timelens.py \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --use_liger True \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${model_path}" \
  --model_id "${model_id}" \
  --conv_type "chatml" \
  --datasets "${datasets}" \
  --remove_unused_columns False \
  --output_dir "${output_dir}" \
  --min_tokens "${min_tokens}" \
  --total_tokens "${total_tokens}" \
  --fps "${fps}" \
  --fps_max_frames "${fps_max_frames}" \
  --target_size "${target_size}" \
  --min_video_len 5 \
  --max_video_len 500 \
  --max_num_words 200 \
  --freeze_vision_tower True \
  --freeze_llm False \
  --freeze_merger False \
  --learning_rate 1e-5 \
  --merger_lr 1e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs "${epochs}" \
  --per_device_train_batch_size "${batch_per_device}" \
  --gradient_accumulation_steps 8 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 2 \
  --dataloader_num_workers 2 \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "${model_id}-sft-test/${run_name}" \
  --logging_dir wandb \
  --save_only_model True \
  --max_steps 20
