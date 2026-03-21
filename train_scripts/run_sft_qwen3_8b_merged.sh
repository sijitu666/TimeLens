#!/usr/bin/env bash

# SFT Training Script - Merged Version
# Combines: L40 GPU optimizations + EG-CoT features
#
# Key features:
# - L40 optimized: total_tokens=7168
# - Supports --prompt_template egcot/legacy
# - Supports --data_path for egcot_jsonl dataset
# - Auto GPU detection

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1

# ==================== Configuration ====================
model_path="./model"
datasets="gemini_refined_data"
model_id="qwen3-vl-8b"

# L40 Optimized Video Processing
min_tokens=64
total_tokens=7168      # L40: reduced from 14336
fps=2
fps_max_frames=""

seed=42

# L40 Optimized Training
global_batch_size=64   # L40: 4 GPUs
batch_per_device=1
num_devices=4          # L40: 4x L40 GPUs
epochs=1
target_size=30000

# DeepSpeed config
deepspeed_config="scripts/zero3_l40.json"
output_root="output/TimeLens-8B/sft"
report_to="none"

# EG-CoT specific parameters
prompt_template="legacy"    # Options: legacy, egcot
data_path=""                  # Required for egcot_jsonl dataset

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
            --datasets) datasets="$2"; shift 2 ;;
            --data_path) data_path="$2"; shift 2 ;;
            --prompt_template) prompt_template="$2"; shift 2 ;;
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
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

# ==================== Main ====================

parse_args "$@"

# Validate model path
if [[ ! -d "${model_path}" ]]; then
    log_error "Model path does not exist: ${model_path}"
    exit 1
fi

# Auto detect GPU count if not specified
if [ -z "${num_devices:-}" ]; then
    num_devices=$(detect_gpu_count)
fi

# Calculate derived parameters
grad_accum_steps=$((global_batch_size / (batch_per_device * num_devices)))
if [[ -z "${fps_max_frames}" ]]; then
    fps_max_frames=$((total_tokens / min_tokens * 2))
fi
run_tag="$(date +%Y%m%d-%H%M)"
prompt_tag="${prompt_template}"
run_name="sft-${run_tag}_PROMPT-${prompt_tag}_MAXFRAMES-${fps_max_frames}_FPS-${fps}_TOTALtokens-${total_tokens}_MINtokens-${min_tokens}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}"

# Print configuration
echo "========================================"
echo "SFT Training Configuration (Merged)"
echo "========================================"
echo "Model path: ${model_path}"
echo "Prompt template: ${prompt_template}"
echo "Total tokens: ${total_tokens} (L40 optimized)"
echo "Global batch size: ${global_batch_size}"
echo "Num devices: ${num_devices}"
echo "Output: ${output_dir}"
echo "========================================"

# Build data path argument
data_path_arg=""
if [[ -n "${data_path}" ]]; then
    data_path_arg="--data_path ${data_path}"
fi

# Launch training with DeepSpeed
deepspeed --num_gpus=${num_devices} training/train/train_sft_timelens.py \
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
  ${data_path_arg} \
  --prompt_template "${prompt_template}" \
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
  --gradient_accumulation_steps "${grad_accum_steps}" \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit "${epochs}" \
  --dataloader_num_workers 4 \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "${model_id}-sft/${run_name}" \
  --logging_dir wandb \
  --save_only_model True
