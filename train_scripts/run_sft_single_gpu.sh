#!/bin/bash
# 单卡 A100 SFT 训练（优先保证能跑起来）

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

model_path="/mnt/bn/aidp-data-multimodal-lf1/zhiwei/LlamaFactory/models/Qwen/Qwen3-VL-8B-Instruct"
output_dir="output/sft_single_gpu"
datasets="gemini_refined_data"
data_path=""
prompt_template=""
num_train_epochs="1"

extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      model_path="$2"; shift 2;;
    --datasets)
      datasets="$2"; shift 2;;
    --output_dir)
      output_dir="$2"; shift 2;;
    --data_path)
      data_path="$2"; shift 2;;
    --prompt_template)
      prompt_template="$2"; shift 2;;
    --num_train_epochs)
      num_train_epochs="$2"; shift 2;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash train_scripts/run_sft_single_gpu.sh \
    [--model_path <path>] \
    [--output_dir <path>] \
    [--datasets <name>] \
    [--data_path <jsonl_or_json>] \
    [--prompt_template <legacy|egcot>] \
    [--num_train_epochs <int>] \
    [<any extra args forwarded to training/train/train_sft_timelens.py>]

Examples:
  bash train_scripts/run_sft_single_gpu.sh \
    --model_path "/path/to/Qwen3-VL-8B-Instruct" \
    --output_dir "output/sft_single_gpu/egcot_run" \
    --datasets egcot_jsonl \
    --data_path "output/TimeLens-8B/sft/egcot_timelens100k_small.jsonl" \
    --prompt_template egcot
EOF
      exit 0;;
    *)
      extra_args+=("$1"); shift;;
  esac
done

if [[ -n "${data_path}" ]]; then
  extra_args+=("--data_path" "${data_path}")
fi
if [[ -n "${prompt_template}" ]]; then
  extra_args+=("--prompt_template" "${prompt_template}")
fi

deepspeed --num_gpus 1 training/train/train_sft_timelens.py \
  --bf16 True \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --use_liger True \
  --deepspeed scripts/zero1.json \
  --model_name_or_path "${model_path}" \
  --model_id "qwen3-vl-8b" \
  --conv_type "chatml" \
  --datasets "${datasets}" \
  --remove_unused_columns False \
  --output_dir "${output_dir}" \
  \
  --bits 4 \
  --lora_enable True \
  --lora_rank 32 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --freeze_vision_tower True \
  --freeze_llm True \
  --freeze_merger False \
  \
  --max_seq_length 8192 \
  --min_tokens 64 \
  --total_tokens 4096 \
  --fps 1 \
  --fps_max_frames 96 \
  --target_size 30000 \
  --min_video_len 5 \
  --max_video_len 500 \
  --max_num_words 200 \
  \
  --learning_rate 2e-5 \
  --merger_lr 1e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs "${num_train_epochs}" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --dataloader_num_workers 2 \
  --seed 42 \
  --report_to none \
  --save_only_model True \
  "${extra_args[@]}"

# NOTE: 本脚本默认是 LoRA/QLoRA（--lora_enable True, --bits 4）。
# training/train/train_sft_timelens.py 会把 LoRA adapter 权重与 processor 保存到 `${output_dir}/lora`。
# 为避免误用（把 adapter 当成完整 checkpoint），这里在训练成功后自动合并成可直接推理/继续训练的完整模型到 `${output_dir}/merged`。
if [[ -f "${output_dir}/lora/adapter_config.json" ]]; then
  python scripts/merge_lora.py \
    --lora_path "${output_dir}/lora" \
    --out_path "${output_dir}/merged" \
    --base_model_path "${model_path}" \
    --skip_if_exists
fi
