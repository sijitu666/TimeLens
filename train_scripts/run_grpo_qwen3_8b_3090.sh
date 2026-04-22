#!/usr/bin/env bash

# GRPO Training Script for RTX 3090 (6 GPUs)
# Optimized for 24GB VRAM per card
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash train_scripts/run_grpo_qwen3_8b_3090.sh \
#     --model_path "path/to/sft/checkpoint" \
#     --raw_anno_path "path/to/filtered/data.jsonl"
#
# Key differences from L40 config:
#   - total_tokens: 3584 (was 7168) - 50% reduction for 24GB VRAM
#   - num_generations: 2 (was 4) - minimum for GRPO, save VRAM
#   - fps_max_frames: 112 (was 224) - auto-calculated
#   - global_batch_size: 12 (was 32-64) - adjusted for 6 cards

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1

# ==================== Configuration ====================
# These should be provided via command line or environment variables
model_path="${MODEL_PATH:-}"  # Required: SFT checkpoint path
raw_anno_path="${RAW_ANNO_PATH:-}"  # Required: filtered data path

# Data configuration
datasets="${DATASETS:-filtered_hybrid}"
model_id="qwen3-vl-8b"

# ==================== 3090 Optimized Video Processing ====================
# Key: Reduce tokens by 50% for 24GB VRAM
min_tokens=64
total_tokens="${TOTAL_TOKENS:-3584}"  # 3090: half of L40's 7168
fps=2
fps_max_frames="${FPS_MAX_FRAMES:-}"  # Auto-calculate: 3584/64*2 = 112

seed="${SEED:-42}"

# ==================== 3090 Optimized GRPO Training ====================
# GRPO specific: num_generations significantly impacts VRAM
# L40 uses 4-8, but 3090 with 24GB should use 2 (minimum for GRPO)
global_batch_size="${GLOBAL_BATCH_SIZE:-12}"  # 6 cards x 2 per card
batch_per_device=1
num_devices="${NUM_DEVICES:-6}"
epochs=1
target_size="${TARGET_SIZE:-2500}"

# GRPO specific parameters
num_generations="${NUM_GENERATIONS:-2}"  # 3090: minimum 2 (L40 uses 4-8)
temperature="${TEMPERATURE:-1.0}"
max_steps="${MAX_STEPS:-1000}"

# Reward functions
reward_funcs="${REWARD_FUNCS:-ear,format}"  # Options: tiou, ear, format

# DeepSpeed config
deepspeed_config="${DEEPSPEED_CONFIG:-scripts/zero3_l40.json}"
output_root="output/TimeLens-8B/grpo"
report_to="none"

# Prompt template
prompt_template="${PROMPT_TEMPLATE:-egcot}"  # Options: legacy, egcot

# LoRA settings
lora_path="${LORA_PATH:-}"
merge_before_train="${MERGE_BEFORE_TRAIN:-true}"
merge_after_train="${MERGE_AFTER_TRAIN:-true}"

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
    local all_same=true
    local first_gpu=""

    # Check nvidia-smi for GPU info
    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        first_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

        # Check if all GPUs are 3090
        local gpu_list=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
        while IFS= read -r gpu; do
            if ! echo "$gpu" | grep -qi "3090"; then
                all_same=false
                log_warn "Non-3090 GPU detected: $gpu"
            fi
        done <<< "$gpu_list"

        if [ "$gpu_count" -gt 0 ]; then
            if [ "$all_same" = true ]; then
                log_info "✓ All $gpu_count GPUs are RTX 3090 - using 3090 optimized config"
            else
                log_warn "Mixed GPU types detected - config may need manual adjustment"
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
            --global_batch_size) global_batch_size="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

# ==================== Main ====================

parse_args "$@"

# Validate required arguments
if [[ -z "${model_path}" ]]; then
    log_error "--model_path is required (use the SFT checkpoint path)"
    log_info "Example: bash train_scripts/run_grpo_qwen3_8b_3090.sh --model_path \"output/.../checkpoint-final\""
    exit 1
fi

if [[ -z "${raw_anno_path}" ]]; then
    log_error "--raw_anno_path is required (use filtering output jsonl path)"
    exit 1
fi

# Auto detect GPU count
num_devices=$(detect_gpu_count)

# Calculate derived parameters
if [[ -z "${fps_max_frames}" ]]; then
    fps_max_frames=$((total_tokens / min_tokens * 2))
fi

grad_accum_steps=$((global_batch_size / (batch_per_device * num_devices)))
if [[ $grad_accum_steps -lt 1 ]]; then
    log_warn "Calculated grad_accum_steps ($grad_accum_steps) < 1, setting to 1"
    grad_accum_steps=1
fi

run_tag="$(date +%Y%m%d-%H%M)"
gpu_tag="3090"
prompt_tag="${prompt_template}"
run_name="grpo-${gpu_tag}-${run_tag}_TOKENS-${total_tokens}_NG-${num_generations}_PROMPT-${prompt_tag}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}"

# Print configuration
echo "========================================"
echo "GRPO Training Configuration (RTX 3090)"
echo "========================================"
echo "Model path: ${model_path}"
echo "Raw anno path: ${raw_anno_path}"
echo "Datasets: ${datasets}"
echo "Prompt template: ${prompt_template}"
echo "Reward funcs: ${reward_funcs}"
echo ""
echo "3090 Optimized Settings:"
echo "  Total tokens: ${total_tokens} (recommended for 24GB VRAM)"
echo "  FPS max frames: ${fps_max_frames}"
echo "  Num generations: ${num_generations} (minimum for GRPO on 3090)"
echo ""
echo "Training Settings:"
echo "  Global batch size: ${global_batch_size}"
echo "  Num devices: ${num_devices}"
echo "  Gradient accum steps: ${grad_accum_steps}"
echo "  Max steps: ${max_steps}"
echo "  Target size: ${target_size}"
echo ""
echo "Output: ${output_dir}"
echo "========================================"

# Confirm before starting
if [[ "${AUTO_CONFIRM:-}" != "true" ]]; then
    read -p "Press Enter to start training, or Ctrl+C to cancel..."
fi

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