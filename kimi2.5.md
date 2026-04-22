# TimeLens项目回顾与下一步指引

## 一、之前实现的内容回顾

### 1. 核心功能实现 (EG-CoT + EAR)

| 模块 | 实现内容 | 关键文件 |
|------|----------|----------|
| **EG-CoT数据构建** | 从TimeLens-100K构建带Evidence-Grounded Chain-of-Thought的SFT数据 | `scripts/build_egcot_data.py` |
| **Prompt模板** | 新增EG-CoT prompt格式（含`\think`/`<answer>`标签） | `training/data/grounding.py` |
| **EAR奖励函数** | Evidence-Aware Reward: R = R_answer × (1+α×R_evidence) × γ_efficiency + δ×𝟙(Perception) | `training/train/reward_funcs.py` |
| **Query复杂度估计** | `QueryComplexityEstimator`支持rule_based/surprisal两种模式 | `training/train/query_complexity.py` |
| **ReasonVTG候选** | 基于TimeLens-Bench生成reasoning-intense候选数据 | `scripts/build_reasonvtg_bench.py` |
| **LoRA自动合并** | 训练前后自动合并LoRA adapter到完整模型 | `scripts/merge_lora.py` |

### 2. 脚本和工具

- **tmux一键启动**: `scripts/tmux_start_egcot_pipeline.sh` (4窗口: 数据准备×2 + SFT + GRPO)
- **CPU数据准备脚本**: `scripts/prepare_egcot_data.sh`, `scripts/prepare_reasonvtg_candidates.sh`
- **训练脚本增强**: 支持自动推断GPU数、自适应batch size、自动合并LoRA

---

## 二、已跑过的实验和结果

### 1. 基线模型评测结果

| 模型 | Charades (R1@0.3/0.5/0.7) | ActivityNet (R1@0.3/0.5/0.7) | QVHighlights (R1@0.3/0.5/0.7) |
|------|---------------------------|-------------------------------|--------------------------------|
| **Qwen3-VL-8B-Instruct** | 68.6 / 52.9 / 27.0 | 62.4 / 51.5 / 35.5 | 74.1 / 63.6 / 48.9 |
| **TimeLens-8B** (官方) | **76.7 / 63.0 / 35.4** | **68.7 / 58.2 / 40.5** | **79.7 / 71.3 / 55.6** |
| **GRPO+EAR** (2026-03-07) | 74.3 / 59.4 / 31.7 | 66.5 / 57.0 / 38.7 | 75.8 / 66.8 / 52.0 |

### 2. 训练记录

| 时间 | 实验类型 | 路径/说明 |
|------|----------|-----------|
| 2026-02-28 | TimeLens-8B官方评测 | `logs/TimeLens-8B_20260228_170750/` |
| 2026-03-05 | Qwen3-VL-8B基线评测 | `logs/Qwen3-VL-8B-Instruct_20260305_192800/` |
| 2026-03-05 | SFT单卡训练(LoRA) | `output/sft_single_gpu/` → 生成`qwen3-merged` |
| 2026-03-07 | GRPO训练(EAR) | `output/TimeLens-8B/grpo/grpo-20260307-2350_*` |
| 2026-03-08 | GRPO合并评测 | `logs/grpo-20260307-*_merged_20260308_124053/` |
| 2026-03-09~12 | 多轮RLVR实验 | `output/TimeLens-8B/rlvr/20260309_*` ~ `20260312_*` |

### 3. 已生成的数据

- **EG-CoT数据**: `output/TimeLens-8B/sft/egcot_timelens100k.jsonl` (29MB, ~30K条)
- **EG-CoT小数据**: `output/TimeLens-8B/sft/egcot_timelens100k_small.jsonl` (79KB)
- **ReasonVTG候选**: `output/reasonvtg_candidates_all.jsonl`

---

## 三、接下来从哪里入手？

### 推荐入手优先级

#### 🔥 第一优先级：跑通EG-CoT SFT（单卡A100）

你已经有了`run_sft_single_gpu.sh`，这是最简单的切入点：

```bash
# 1. 激活环境
source /home/tiger/miniconda3/etc/profile.d/conda.sh
conda activate timelens
cd /mnt/bn/aidp-data-multimodal-lf1/zhiwei/TimeLens

# 2. 跑EG-CoT SFT（使用已生成的小数据集快速测试）
bash train_scripts/run_sft_single_gpu.sh \
  --model_path "/mnt/bn/aidp-data-multimodal-lf1/zhiwei/LlamaFactory/models/Qwen/Qwen3-VL-8B-Instruct" \
  --datasets egcot_jsonl \
  --data_path "output/TimeLens-8B/sft/egcot_timelens100k_small.jsonl" \
  --prompt_template egcot
```

#### 🔥 第二优先级：GRPO训练（单卡A100）

```bash
# 使用已有的SFT checkpoint或刚才训练的LoRA
bash train_scripts/run_grpo_and_eval_qwen3_8b.sh \
  --model_path "/mnt/bn/aidp-data-multimodal-lf1/zhiwei/TimeLens/output/sft_single_gpu/qwen3-merged" \
  --raw_anno_path "/mnt/bn/aidp-data-multimodal-lf1/zhiwei/TimeLens/output/TimeLens-8B/filter-data/prebuilt/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl" \
  --prompt_template egcot \
  --reward_funcs ear
```

#### 第三优先级：数据准备（CPU任务，可以后台跑）

```bash
# 如果还没生成EG-CoT数据
bash scripts/prepare_egcot_data.sh \
  --input_jsonl data/TimeLens-100K/timelens-100k.jsonl \
  --output_jsonl output/TimeLens-8B/sft/egcot_timelens100k.jsonl \
  --target_reasoning_ratio 0.4
```

---

## 四、修改的代码详细说明

### 已修改的核心文件（12个文件，+602/-45行）

| 文件 | 修改内容 |
|------|----------|
| `training/data/grounding.py` | 新增`PROMPT_EGCOT`模板；支持`egcot_jsonl`数据集；`--prompt_template`开关 |
| `training/data/hybrid.py` | datasets列表新增`egcot_jsonl` |
| `training/params.py` | DataArguments新增`prompt_template`参数 |
| `training/train/reward_funcs.py` | 新增`ear`奖励函数；`format_reward`放宽支持感知型答案 |
| `training/train/query_complexity.py` | 新增`QueryComplexityEstimator`类（rule_based/surprisal） |
| `training/train/train_sft_timelens.py` | 支持`prompt_template`参数透传 |
| `training/train/train_utils.py` | 支持新数据格式 |
| `evaluation/compute_metrics.py` | 新增`--group_by_query_type`按query类型分组统计 |
| `evaluation/eval_dataloader.py` | 修复model load支持`trust_remote_code`和LoRA adapter加载 |
| `train_scripts/run_grpo_qwen3_8b.sh` | 自动推断GPU数、自适应batch size、自动合并LoRA、新增参数透传 |
| `train_scripts/run_grpo_and_eval_qwen3_8b.sh` | 同上 + 训练后自动合并+评测 |
| `README.md` | 更新使用说明 |

### 新增文件（10个）

| 文件 | 用途 |
|------|------|
| `scripts/build_egcot_data.py` | 构建EG-CoT SFT数据 |
| `scripts/build_reasonvtg_bench.py` | 构建ReasonVTG候选 |
| `scripts/prepare_egcot_data.sh` | 一键EG-CoT数据准备 |
| `scripts/prepare_reasonvtg_candidates.sh` | 一键ReasonVTG候选准备 |
| `scripts/merge_lora.py` | LoRA权重合并工具 |
| `scripts/tmux_start_egcot_pipeline.sh` | tmux一键启动4窗口 |
| `IMPLEMENTATION_LOG_EGCOT_EAR_20260310.md` | 实现日志（详细） |
| `OPERATION_LOG_20260309.md` | 操作日志 |
| `evidence-grounded-tvg.SKILL.md` | 技能沉淀文档 |
| `training/train/query_complexity.py` | Query复杂度估计器 |

---

## 五、单卡A100测试推荐命令（总结）

### 最简单的入手（强烈推荐先跑这个）：

```bash
# 1. 进入环境
source /home/tiger/miniconda3/etc/profile.d/conda.sh
conda activate timelens
cd /mnt/bn/aidp-data-multimodal-lf1/zhiwei/TimeLens

# 2. 用已有的小数据集快速测试EG-CoT SFT（~10分钟）
bash train_scripts/run_sft_single_gpu.sh \
  --model_path "/mnt/bn/aidp-data-multimodal-lf1/zhiwei/LlamaFactory/models/Qwen/Qwen3-VL-8B-Instruct" \
  --datasets egcot_jsonl \
  --data_path "output/TimeLens-8B/sft/egcot_timelens100k_small.jsonl" \
  --prompt_template egcot
```

### 如果想直接跑完整流程：

```bash
# tmux一键启动（4窗口并行：数据准备 + SFT + GRPO）
bash scripts/tmux_start_egcot_pipeline.sh \
  --base_model_path "/mnt/bn/aidp-data-multimodal-lf1/zhiwei/LlamaFactory/models/Qwen/Qwen3-VL-8B-Instruct" \
  --filtered_jsonl "/mnt/bn/aidp-data-multimodal-lf1/zhiwei/TimeLens/output/TimeLens-8B/filter-data/prebuilt/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl" \
  --run_data_prep true \
  --conda_env timelens
```

---

*文档生成时间: 2026-03-14*
