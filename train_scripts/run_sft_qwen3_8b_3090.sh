#!/usr/bin/env bash

# SFT Training Script for RTX 3090 (6 GPUs)
# Optimized for 24GB VRAM per card
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash train_scripts/run_sft_qwen3_8b_3090.sh
#
# Key differences from L40 config:
#   - total_tokens: 3584 (was 7168) - 50% reduction for 24GB VRAM
#   - fps_max_frames: 112 (was 224) - auto-calculated
#   - global_batch_size: 24 (was 64) - adjusted for 6 cards
#   - gradient_accumulation: adjusted accordingly

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1

# ==================== Configuration ====================
# Model path - modify this to your actual model path
model_path="${MODEL_PATH:-./model}"

# Data configuration
datasets="${DATASETS:-egcot_jsonl}"
data_path="${DATA_PATH:-}"  # Required for egcot_jsonl

# Model settings
model_id="qwen3-vl-8b"

# ==================== 3090 Optimized Video Processing ====================
# Key: Reduce tokens by 50% for 24GB VRAM
min_tokens=64
total_tokens="${TOTAL_TOKENS:-3584}"  # 3090: half of L40's 7168
fps=2
fps_max_frames="${FPS_MAX_FRAMES:-}"  # Auto-calculate: 3584/64*2 = 112

seed="${SEED:-42}"

# ==================== 3090 Optimized Training ====================
# Adjust batch size for 6x 3090 cards
global_batch_size="${GLOBAL_BATCH_SIZE:-24}"  # 6 cards x 4 per card
batch_per_device=1
num_devices="${NUM_DEVICES:-6}"
epochs=1
target_size="${TARGET_SIZE:-30000}"

# DeepSpeed config - use ZeRO-3 for 24GB VRAM
deepspeed_config="${DEEPSPEED_CONFIG:-scripts/zero3_l40.json}"
output_root="output/TimeLens-8B/sft"
report_to="none"

# EG-CoT specific parameters
prompt_template="${PROMPT_TEMPLATE:-egcot}"  # Options: legacy, egcot

# ==================== Functions ====================

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

log_warn() {
    echo "[WARN] $1"
}

# Auto detect GPU count and verify 3090
detect_gpu_count() {
    local gpu_count=0
    local gpu_type=""

    # Check nvidia-smi for GPU info
    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

        if [ "$gpu_count" -gt 0 ]; then
            log_info "Detected $gpu_count GPU(s): $gpu_type"

            # Check if it's 3090
            if echo "$gpu_type" | grep -qi "3090"; then
                log_info "✓ RTX 3090 detected - using 3090 optimized config"
            else
                log_warn "GPU is not RTX 3090 ($gpu_type) - config may need adjustment"
            fi
        fi
    fi

    # Also check CUDA_VISIBLE_DEVICES
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        local cuda_count=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        log_info "CUDA_VISIBLE_DEVICES set to use $cuda_count GPU(s)"
        gpu_count=$cuda_count
    fi

    if [ "$gpu_count" -eq 0 ]; then
        log_warn "Could not detect GPU count, using default: $num_devices"
        gpu_count=$num_devices
    fi

    echo $gpu_count
}

# Validate configuration
validate_config() {
    local errors=0

    # Check model path
    if [[ ! -d "${model_path}" ]]; then
        log_error "Model path does not exist: ${model_path}"
        log_info "Please set MODEL_PATH environment variable or modify the script"
        ((errors++))
    fi

    # Check data_path for egcot_jsonl
    if [[ "$datasets" == "egcot_jsonl" && -z "$data_path" ]]; then
        log_error "Using egcot_jsonl dataset but DATA_PATH is not set"
        log_info "Please set DATA_PATH to the egcot jsonl file path"
        ((errors++))
    fi

    # Warn about total_tokens for 3090
    if [[ "$total_tokens" -gt 4096 ]]; then
        log_warn "total_tokens=$total_tokens may be too high for RTX 3090 (24GB)"
        log_info "Recommended: 3584 or 4096 for 3090"
    fi

    return $errors
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model_path) model_path="$2"; shift 2 ;;
            --data_path) data_path="$2"; shift 2 ;;
            --datasets) datasets="$2"; shift 2 ;;
            --prompt_template) prompt_template="$2"; shift 2 ;;
            --total_tokens) total_tokens="$2"; shift 2 ;;
            --global_batch_size) global_batch_size="$2"; shift 2 ;;
            --target_size) target_size="$2"; shift 2 ;;
            --num_devices) num_devices="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

# ==================== Main ====================

parse_args "$@"

# Auto detect GPU count
if [ -z "${num_devices:-}" ] || [ "$num_devices" -eq 6 ]; then
    detected_devices=$(detect_gpu_count)
    if [ "$detected_devices" -ne "$num_devices" ]; then
        log_info "Updating num_devices from $num_devices to $detected_devices"
        num_devices=$detected_devices
    fi
fi

# Validate configuration
if ! validate_config; then
    log_error "Configuration validation failed. Please fix the errors above."
    exit 1
fi

# Calculate derived parameters
if [[ -z "${fps_max_frames}" ]]; then
    fps_max_frames=$((total_tokens / min_tokens * 2))
fi

grad_accum_steps=$((global_batch_size / (batch_per_device * num_devices)))
run_tag="$(date +%Y%m%d-%H%M)"
gpu_tag="3090"
prompt_tag="${prompt_template}"
run_name="sft-${gpu_tag}-${run_tag}_TOKENS-${total_tokens}_PROMPT-${prompt_tag}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}"

# Build data path argument
data_path_arg=""
if [[ -n "${data_path}" ]]; then
    data_path_arg="--data_path ${data_path}"
fi

# Print configuration
echo "========================================"
echo "SFT Training Configuration (RTX 3090)"
echo "========================================"
echo "Model path: ${model_path}"
echo "Datasets: ${datasets}"
echo "Data path: ${data_path:-<not set>}"
echo "Prompt template: ${prompt_template}"
echo "Total tokens: ${total_tokens} (3090 optimized)"
echo "FPS max frames: ${fps_max_frames}"
echo "Global batch size: ${global_batch_size}"
echo "Num devices: ${num_devices}"
echo "Gradient accum steps: ${grad_accum_steps}"
echo "Target size: ${target_size}"
echo "Output: ${output_dir}"
echo "========================================"

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