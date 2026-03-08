#!/usr/bin/env bash

# GRPO Training Script for L40 GPUs (4x L40 48GB)
# Adjusted from original H20 config to fit L40 memory constraints

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1

model_path=""
raw_anno_path=""
datasets="filtered_hybrid"
model_id="qwen3-vl-8b"

# ============ L40 Adjusted Video Processing Parameters ============
# Original: total_tokens=14336
# L40 Adjustment: Reduce to 7168 to fit within 48GB
min_tokens=64
total_tokens=7168
fps=2
fps_max_frames=""

seed=42

# ============ L40 Adjusted GRPO Training Parameters ============
# Original H20 config: global_batch_size=64, num_devices=8, num_generations=8
# L40 Adjustment: Reduce batch size, generations, and use 4 GPUs
# GRPO is memory intensive because it maintains multiple generations
global_batch_size=32         # Reduced from 64
batch_per_device=1
num_devices=4                # 4x L40 GPUs
epochs=1
target_size=2500

# Use ZeRO-3 with offloading for GRPO (more memory efficient)
deepspeed_config="scripts/zero3_l40.json"
output_root="output/TimeLens-8B/grpo"
report_to="none"

# GRPO-specific parameters (L40 optimized)
num_generations=4            # Reduced from 8 (major memory saver)
temperature=1.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --raw_anno_path) raw_anno_path="$2"; shift 2 ;;
    --datasets) datasets="$2"; shift 2 ;;
    --min_tokens) min_tokens="$2"; shift 2 ;;
    --total_tokens) total_tokens="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --fps_max_frames) fps_max_frames="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --batch_per_device) batch_per_device="$2"; shift 2 ;;
    --num_devices) num_devices="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --target_size) target_size="$2"; shift 2 ;;
    --deepspeed_config) deepspeed_config="$2"; shift 2 ;;
    --output_root) output_root="$2"; shift 2 ;;
    --report_to) report_to="$2"; shift 2 ;;
    --num_generations) num_generations="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "${model_path}" ]]; then
  echo "Error: --model_path is required (use the SFT checkpoint path)."
  exit 1
fi

if [[ -z "${raw_anno_path}" ]]; then
  echo "Error: --raw_anno_path is required (use filtering output jsonl path)."
  exit 1
fi

if [[ ! -d "${model_path}" ]]; then
  echo "Error: Model path does not exist: ${model_path}"
  exit 1
fi

if [[ ! -f "${raw_anno_path}" ]]; then
  echo "Error: Raw annotation path does not exist: ${raw_anno_path}"
  exit 1
fi

echo "========================================"
echo "L40 GRPO Training Configuration"
echo "========================================"
echo "Model path: ${model_path}"
echo "Raw anno path: ${raw_anno_path}"
echo "Total tokens: ${total_tokens} (L40 optimized)"
echo "Num generations: ${num_generations} (L40 optimized)"
echo "Global batch size: ${global_batch_size}"
echo "Num devices: ${num_devices}"
echo "Deepspeed config: ${deepspeed_config}"
echo "========================================"

# Calculate derived parameters
grad_accum_steps=$((global_batch_size / (batch_per_device * num_devices)))
if [[ -z "${fps_max_frames}" ]]; then
  fps_max_frames=$((total_tokens / min_tokens * 2))
fi
run_tag="$(date +%Y%m%d-%H%M)"
run_name="grpo-${run_tag}_MAXFRAMES-${fps_max_frames}_FPS-${fps}_TOTALtokens-${total_tokens}_MINtokens-${min_tokens}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}"
echo "Output directory: ${output_dir}"

# Launch training with DeepSpeed
deepspeed --num_gpus=${num_devices} training/train/train_grpo_timelens.py \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${model_path}" \
  --model_id "${model_id}" \
  --datasets "${datasets}" \
  --raw_anno_path "${raw_anno_path}" \
  --fixed_gaussian_sampling True \
  --gaussian_filter_mean 0.05 \
  --gaussian_filter_std 0.2 \
  --target_size "${target_size}" \
  --remove_unused_columns False \
  --output_dir "${output_dir}" \
  --min_tokens "${min_tokens}" \
  --total_tokens "${total_tokens}" \
  --fps "${fps}" \
  --fps_max_frames "${fps_max_frames}" \
  --min_video_len 5 \
  --max_video_len 500 \
  --max_num_words 200 \
  --freeze_vision_tower True \
  --freeze_llm False \
  --freeze_merger False \
  --lr_scheduler_type constant \
  --learning_rate 1e-6 \
  --num_train_epochs "${epochs}" \
  --per_device_train_batch_size "${batch_per_device}" \
  --gradient_accumulation_steps "${grad_accum_steps}" \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 5 \
  --dataloader_num_workers 4 \
  --log_completions True \
  --use_liger False \
  --use_liger_loss False \
  --reward_funcs tiou \
  --num_generations "${num_generations}" \
  --steps_per_generation 1 \
  --temperature 1.0 \
  --scale_rewards False \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "${model_id}-grpo/${run_name}" \
  --logging_dir wandb \
  --save_only_model True \
  --max_steps 100
