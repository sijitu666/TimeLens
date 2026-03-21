# EG-CoT (Evidence-Grounded Chain-of-Thought) 与 EAR (Evidence-Aware Reward) 完整使用指南

**版本**: v1.0
**日期**: 2026-03-19
**适用**: TimeLens项目L40 GPU环境

---

## 📋 目录

1. [快速开始](#1-快速开始)
2. [EG-CoT数据构建](#2-eg-cot数据构建)
3. [SFT训练](#3-sft训练)
4. [GRPO训练与EAR](#4-grpo训练与ear)
5. [评测与分析](#5-评测与分析)
6. [完整工作流示例](#6-完整工作流示例)

---

## 1. 快速开始

### 1.1 环境检查

```bash
# 检查GPU
nvidia-smi

# 检查模型权重
ls -lh ./model/

# 检查数据路径
ls -lh data/TimeLens-100K/
ls -lh data/TimeLens-Bench/
```

### 1.2 已合并文件清单

**核心代码修改 (已自动合并)**:
- ✅ `training/data/grounding.py` - EG-CoT Prompt支持
- ✅ `training/data/hybrid.py` - egcot_jsonl数据集支持
- ✅ `training/params.py` - prompt_template参数
- ✅ `training/train/reward_funcs.py` - EAR奖励函数
- ✅ `evaluation/compute_metrics.py` - Query类型分组统计
- ✅ `evaluation/eval_dataloader.py` - LoRA支持

**新增脚本 (已自动添加)**:
- ✅ `scripts/build_egcot_data.py` - EG-CoT数据构建
- ✅ `scripts/build_reasonvtg_bench.py` - ReasonVTG评测构建
- ✅ `scripts/merge_lora.py` - LoRA权重合并
- ✅ `scripts/prepare_egcot_data.sh` - 数据准备封装
- ✅ `scripts/prepare_reasonvtg_candidates.sh` - 评测准备封装

**统一运行脚本 (已创建)**:
- ✅ `train_scripts/run_sft_qwen3_8b_merged.sh` - SFT统一脚本
- ✅ `train_scripts/run_grpo_qwen3_8b_merged.sh` - GRPO统一脚本

---

## 2. EG-CoT数据构建

### 2.1 快速构建EG-CoT数据

```bash
# 方式1: 使用启发式方法（不依赖外部API，推荐先用这个）
python scripts/build_egcot_data.py \
  --input_jsonl data/TimeLens-100K/timelens-100k.jsonl \
  --output_jsonl output/egcot_data/egcot_timelens100k.jsonl \
  --llm_provider none \
  --target_reasoning_ratio 0.4

# 方式2: 使用OpenAI GPT-4生成推理链（需要API key）
export OPENAI_API_KEY="your-api-key"
python scripts/build_egcot_data.py \
  --input_jsonl data/TimeLens-100K/timelens-100k.jsonl \
  --output_jsonl output/egcot_data/egcot_timelens100k_gpt4.jsonl \
  --llm_provider openai \
  --llm_model gpt-4o-mini \
  --target_reasoning_ratio 0.4

# 方式3: 使用Gemini生成推理链
export GOOGLE_API_KEY="your-api-key"
python scripts/build_egcot_data.py \
  --input_jsonl data/TimeLens-100K/timelens-100k.jsonl \
  --output_jsonl output/egcot_data/egcot_timelens100k_gemini.jsonl \
  --llm_provider gemini \
  --llm_model gemini-1.5-flash \
  --target_reasoning_ratio 0.4
```

### 2.2 使用封装脚本

```bash
# 后台运行数据构建（CPU任务）
bash scripts/prepare_egcot_data.sh \
  data/TimeLens-100K/timelens-100k.jsonl \
  output/egcot_data/egcot_timelens100k.jsonl

# 查看日志
tail -f logs/prepare_egcot_data_*.log
```

---

## 3. SFT训练

### 3.1 传统SFT训练（legacy prompt）

```bash
# 4卡L40训练 - 使用传统TimeLens prompt
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train_scripts/run_sft_qwen3_8b_merged.sh \
  --model_path "./model" \
  --datasets "gemini_refined_data" \
  --prompt_template "legacy" \
  --total_tokens 7168 \
  --global_batch_size 64
```

### 3.2 EG-CoT SFT训练

```bash
# 4卡L40训练 - 使用EG-CoT prompt
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train_scripts/run_sft_qwen3_8b_merged.sh \
  --model_path "./model" \
  --datasets "egcot_jsonl" \
  --data_path "output/egcot_data/egcot_timelens100k.jsonl" \
  --prompt_template "egcot" \
  --total_tokens 7168 \
  --global_batch_size 64
```

### 3.3 单卡测试

```bash
# 单卡L40冒烟测试 - EG-CoT
CUDA_VISIBLE_DEVICES=0 \
bash train_scripts/run_sft_qwen3_8b_merged.sh \
  --model_path "./model" \
  --datasets "egcot_jsonl" \
  --data_path "output/egcot_data/egcot_timelens100k.jsonl" \
  --prompt_template "egcot" \
  --total_tokens 3584 \
  --global_batch_size 8 \
  --target_size 100
```

---

## 4. GRPO训练与EAR

### 4.1 传统GRPO训练（tiou奖励）

```bash
# 4卡L40 GRPO训练 - 使用tIoU奖励
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train_scripts/run_grpo_qwen3_8b_merged.sh \
  --model_path "output/TimeLens-8B/sft/YOUR_SFT_RUN/checkpoint-XXX" \
  --raw_anno_path "output/TimeLens-8B/filter-data/YOUR_FILTER_RUN/gemini_refined_data.jsonl" \
  --prompt_template "legacy" \
  --reward_funcs "tiou" \
  --total_tokens 7168 \
  --num_generations 4 \
  --max_steps 1000
```

### 4.2 EAR (Evidence-Aware Reward) GRPO训练

```bash
# 4卡L40 GRPO训练 - 使用EAR奖励（EG-CoT推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train_scripts/run_grpo_qwen3_8b_merged.sh \
  --model_path "output/TimeLens-8B/sft/YOUR_EGCOT_SFT_RUN/checkpoint-XXX" \
  --raw_anno_path "output/TimeLens-8B/filter-data/YOUR_FILTER_RUN/gemini_refined_data.jsonl" \
  --prompt_template "egcot" \
  --reward_funcs "ear,format" \
  --total_tokens 7168 \
  --num_generations 4 \
  --max_steps 1000
```

### 4.3 组合奖励函数

```bash
# 使用多种奖励函数组合
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train_scripts/run_grpo_qwen3_8b_merged.sh \
  --model_path "output/TimeLens-8B/sft/YOUR_SFT_RUN/checkpoint-XXX" \
  --raw_anno_path "output/TimeLens-8B/filter-data/YOUR_FILTER_RUN/gemini_refined_data.jsonl" \
  --prompt_template "egcot" \
  --reward_funcs "ear,tiou,format" \
  --total_tokens 7168 \
  --num_generations 4 \
  --max_steps 1000
```

---

## 5. 评测与分析

### 5.1 基础评测

```bash
# 单卡评测 - 基础命令
CUDA_VISIBLE_DEVICES=0 \
model_path="output/TimeLens-8B/sft/YOUR_RUN/checkpoint-final" \
total_tokens=7168 \
bash scripts/eval_timelens_bench.sh

# 指定特定checkpoint评测
CUDA_VISIBLE_DEVICES=0 \
model_path="output/TimeLens-8B/sft/sft-20260319-1200_PROMPT-egcot_.../checkpoint-1000" \
total_tokens=7168 \
bash scripts/eval_timelens_bench.sh
```

### 5.2 Query类型分组评测

```bash
# 评测并查看Query类型分组统计
python evaluation/compute_metrics.py \
  -f logs/YOUR_MODEL/charades-timelens.jsonl \
  --group_by_query_type

# 示例输出：
# --- Grouped by query_type ---
# perception: N=312 | IOU0.3=72.12 IOU0.5=61.54 IOU0.7=45.83 mIOU=55.21
# causal: N=89 | IOU0.3=65.17 IOU0.5=54.49 IOU0.7=38.20 mIOU=48.67
# comparison: N=45 | IOU0.3=58.23 IOU0.5=47.89 IOU0.7=32.44 mIOU=42.15
```

### 5.3 LoRA模型评测

```bash
# 如果模型是LoRA adapter，先合并再评测
python scripts/merge_lora.py \
  --lora_path "output/TimeLens-8B/grpo/YOUR_RUN/checkpoint-1000" \
  --out_path "output/TimeLens-8B/grpo/YOUR_RUN/checkpoint-1000_merged" \
  --base_model_path "./model"

# 然后评测合并后的模型
CUDA_VISIBLE_DEVICES=0 \
model_path="output/TimeLens-8B/grpo/YOUR_RUN/checkpoint-1000_merged" \
total_tokens=7168 \
bash scripts/eval_timelens_bench.sh
```

---

## 6. 完整工作流示例

### 6.1 EG-CoT + EAR 完整流程

```bash
# ========== Step 1: 构建EG-CoT数据 ==========
python scripts/build_egcot_data.py \
  --input_jsonl data/TimeLens-100K/timelens-100k.jsonl \
  --output_jsonl output/egcot_data/egcot_timelens100k.jsonl \
  --llm_provider none \
  --target_reasoning_ratio 0.4

# ========== Step 2: EG-CoT SFT训练 ==========
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train_scripts/run_sft_qwen3_8b_merged.sh \
  --model_path "./model" \
  --datasets "egcot_jsonl" \
  --data_path "output/egcot_data/egcot_timelens100k.jsonl" \
  --prompt_template "egcot" \
  --total_tokens 7168 \
  --global_batch_size 64

# ========== Step 3: 数据过滤（生成用于GRPO的数据） ==========
# 使用SFT后的模型对全量数据进行过滤推理
# 这里需要运行过滤脚本...

# ========== Step 4: EAR GRPO训练 ==========
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train_scripts/run_grpo_qwen3_8b_merged.sh \
  --model_path "output/TimeLens-8B/sft/YOUR_EGCOT_RUN/checkpoint-final" \
  --raw_anno_path "output/TimeLens-8B/filter-data/YOUR_FILTER_RUN/gemini_refined_data.jsonl" \
  --prompt_template "egcot" \
  --reward_funcs "ear,format" \
  --total_tokens 7168 \
  --num_generations 4 \
  --max_steps 1000

# ========== Step 5: 评测 ==========
# 合并LoRA（如果使用）
python scripts/merge_lora.py \
  --lora_path "output/TimeLens-8B/grpo/YOUR_GRPO_RUN/checkpoint-1000" \
  --out_path "output/TimeLens-8B/grpo/YOUR_GRPO_RUN/checkpoint-1000_merged"

# 评测
CUDA_VISIBLE_DEVICES=0 \
model_path="output/TimeLens-8B/grpo/YOUR_GRPO_RUN/checkpoint-1000_merged" \
total_tokens=7168 \
bash scripts/eval_timelens_bench.sh

# 查看Query类型分组统计
python evaluation/compute_metrics.py \
  -f logs/YOUR_MODEL/charades-timelens.jsonl \
  --group_by_query_type
```

### 6.2 关键超参数建议

| 阶段 | 参数 | Legacy | EG-CoT |
|------|------|--------|--------|
| SFT | prompt_template | legacy | egcot |
| SFT | total_tokens | 7168 | 7168 |
| GRPO | reward_funcs | tiou | ear,format |
| GRPO | num_generations | 4 | 4 |
| GRPO | prompt_template | legacy | egcot |

---

## 📚 参考文档

- 打包文档: `pack_20260319_124654/docs/`
- 原始README: `README.md`
- L40训练指南: `L40_TRAINING_GUIDE.md`

---

**维护者**: Claude Code
**最后更新**: 2026-03-19
