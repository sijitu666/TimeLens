#!/usr/bin/env bash

# GRPO Training Script - Merged Version
# Combines: L40 GPU optimizations + EG-CoT/RLVR features from pack
#
# Key features:
# - L40 optimized: total_tokens=7168, num_generations=4
# - Supports --prompt_template egcot/legacy
# - Supports --reward_funcs tiou/ear/format
# - Auto LoRA merge before/after training
# - Auto GPU detection and adaptive batch size

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1

# ==================== Configuration ====================
model_path=""
raw_anno_path=""
datasets="filtered_hybrid"
model_id="qwen3-vl-8b"

# L40 Optimized Video Processing
min_tokens=64
total_tokens=7168      # L40: reduced from 14336
fps=2
fps_max_frames=""

seed=42

# L40 Optimized GRPO Training
global_batch_size=32   # L40: reduced from 64
batch_per_device=1
num_devices=4          # L40: 4x L40 GPUs
epochs=1
target_size=2500

# DeepSpeed config
deepspeed_config="scripts/zero3_l40.json"
output_root="output/TimeLens-8B/grpo"
report_to="none"

# GRPO-specific (L40 optimized)
num_generations=4      # L40: reduced from 8 (major memory saver)
temperature=1.0
max_steps=100          # Can be increased for full training

# EG-CoT/RLVR specific parameters (NEW from pack)
prompt_template="legacy"    # Options: legacy, egcot
reward_funcs="tiou"           # Options: tiou, ear, format (comma-separated)

# LoRA settings
lora_path=""
merge_before_train=true
merge_after_train=true

# Auto-infer num_devices from CUDA_VISIBLE_DEVICES or nvidia-smi
auto_detect_gpu=true

# ==================== Functions ====================

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

# Auto detect GPU count
detect_gpu_count() {
    local gpu_count=0
    
    # Method 1: Check CUDA_VISIBLE_DEVICES
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        log_info "Detected $gpu_count GPU(s) from CUDA_VISIBLE_DEVICES"
        echo $gpu_count
        return
    fi
    
    # Method 2: Check nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            log_info "Detected $gpu_count GPU(s) from nvidia-smi"
            echo $gpu_count
            return
        fi
    fi
    
    # Fallback
    log_info "Could not detect GPU count, using default: $num_devices"
    echo $num_devices
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model_path) model_path="$2"; shift 2 ;;
            --raw_anno_path) raw_anno_path="$2"; shift 2 ;;
            --datasets) datasets="$2"; shift 2 ;;
            --prompt_template) prompt_template="$2"; shift 2 ;;
            --reward_funcs) reward_funcs="$2"; shift 2 ;;
            --lora_path) lora_path="$2"; shift 2 ;;
            --total_tokens) total_tokens="$2"; shift 2 ;;
            --num_generations) num_generations="$2"; shift 2 ;;
            --max_steps) max_steps="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

# ==================== Main ====================

parse_args "$@"

# Validate required arguments
if [[ -z "${model_path}" ]]; then
    log_error "--model_path is required (use the SFT checkpoint path)"
    exit 1
fi

if [[ -z "${raw_anno_path}" ]]; then
    log_error "--raw_anno_path is required (use filtering output jsonl path)"
    exit 1
fi

# Auto detect GPU count if enabled
if [ "$auto_detect_gpu" = true ]; then
    num_devices=$(detect_gpu_count)
fi

# Calculate derived parameters
grad_accum_steps=$((global_batch_size / (batch_per_device * num_devices)))
if [[ -z "${fps_max_frames}" ]]; then
    fps_max_frames=$((total_tokens / min_tokens * 2))
fi
run_tag="$(date +%Y%m%d-%H%M)"
run_name="grpo-${run_tag}_MAXFRAMES-${fps_max_frames}_FPS-${fps}_TOTALtokens-${total_tokens}_MINtokens-${min_tokens}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}"

# Print configuration
echo "========================================"
echo "GRPO Training Configuration (Merged)"
echo "========================================"
echo "Model path: ${model_path}"
echo "Prompt template: ${prompt_template}"
echo "Reward funcs: ${reward_funcs}"
echo "Total tokens: ${total_tokens} (L40 optimized)"
echo "Num generations: ${num_generations}"
echo "Global batch size: ${global_batch_size}"
echo "Num devices: ${num_devices}"
echo "Max steps: ${max_steps}"
echo "Output: ${output_dir}"
echo "========================================"

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
  --reward_funcs "${reward_funcs}" \
  --num_generations "${num_generations}" \
  --steps_per_generation 1 \
  --temperature 1.0 \
  --scale_rewards False \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "${model_id}-grpo/${run_name}" \
  --logging_dir wandb \
  --save_only_model True \
  --max_steps "${max_steps}" \
  --prompt_template "${prompt_template}"
