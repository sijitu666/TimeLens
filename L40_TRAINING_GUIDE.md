# L40 GPU 训练配置指南

## 概述

本指南帮助你在 **NVIDIA L40 (48GB)** 显卡上运行 TimeLens 的训练和评估。

相比原作者使用的 **H20 (96GB)**，L40 的显存只有一半，需要调整以下关键参数。

---

## 关键参数调整说明

| 参数 | 原始 (H20) | L40 推荐 | 说明 |
|------|-----------|----------|------|
| `total_tokens` | 14336 | 7168 (50%) | 降低视频编码token数 |
| `num_generations` | 8 | 4 (50%) | GRPO生成数，对显存影响大 |
| `global_batch_size` | 128/64 | 64/32 (50%) | 根据GPU数量调整 |
| `fps_max_frames` | 448 | 224 | 视频最大帧数 |
| `num_devices` | 8 | 4 | L40 GPU数量 |

---

## 生成的配置文件

### 1. SFT 训练脚本
- **完整4卡训练**: `train_scripts/run_sft_qwen3_8b_l40.sh`
- **单卡测试**: `train_scripts/run_sft_qwen3_8b_l40_test.sh`

### 2. GRPO 训练脚本
- **完整4卡训练**: `train_scripts/run_grpo_qwen3_8b_l40.sh`

### 3. DeepSpeed 配置
- **L40专用**: `scripts/zero3_l40.json`
  - 启用了 CPU offloading 节省显存

---

## 快速开始

### 第一步：环境检查

```bash
# 检查CUDA和GPU
nvidia-smi

# 确认模型权重存在
ls -lh ./model/

# 测试PyTorch能识别GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

---

### 第二步：评估测试（推荐先跑）

评估比训练省显存，建议先测试评估流程是否正常。

```bash
# ============ 单卡L40评估（total_tokens=7168） ============
# 预估显存: ~25GB

CUDA_VISIBLE_DEVICES=0 \
model_path="./model" \
total_tokens=7168 \
bash scripts/eval_timelens_bench.sh
```

```bash
# ============ 如果显存仍不足，用更保守配置 ============
# 预估显存: ~15GB

CUDA_VISIBLE_DEVICES=0 \
model_path="./model" \
total_tokens=3584 \
FPS=1 \
bash scripts/eval_timelens_bench.sh
```

---

### 第三步：SFT训练测试（单卡冒烟测试）

在完整4卡训练前，强烈建议先做单卡冒烟测试。

```bash
# ============ 单卡L40 SFT冒烟测试 ============
# 使用最小配置，快速验证代码能跑通
# 预估显存: ~35GB
# 运行时间: 几分钟

CUDA_VISIBLE_DEVICES=0 \
bash train_scripts/run_sft_qwen3_8b_l40_test.sh --model_path "./model"
```

如果成功运行，你会看到：
- 模型加载成功
- 数据加载成功
- 训练步骤开始执行
- 显存占用在35-40GB之间

---

### 第四步：完整4卡SFT训练

冒烟测试通过后，开始正式4卡训练。

```bash
# ============ 4卡L40 SFT完整训练 ============
# 配置: total_tokens=7168, batch_size=64
# 预估显存/卡: ~40-42GB
# 预计训练时间: 取决于数据量

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train_scripts/run_sft_qwen3_8b_l40.sh --model_path "./model"
```

训练输出目录：`output/TimeLens-8B/sft/sft-YYYYMMDD-HHMM_.../`

---

### 第五步：GRPO训练（可选，需要完成SFT和过滤）

GRPO需要SFT后的模型，以及过滤后的数据。

```bash
# ============ 4卡L40 GRPO训练 ============
# 注意: 需要先完成SFT和过滤阶段

# 假设SFT输出在: output/TimeLens-8B/sft/sft-20250101-1200_.../
# 假设过滤输出在: output/TimeLens-8B/filter-data/filter-20250101-1300_.../gemini_refined_data.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train_scripts/run_grpo_qwen3_8b_l40.sh \
  --model_path "output/TimeLens-8B/sft/YOUR_SFT_RUN_DIR" \
  --raw_anno_path "output/TimeLens-8B/filter-data/YOUR_FILTER_RUN_DIR/gemini_refined_data.jsonl"
```

---

## 故障排查

### 1. OOM (显存不足)

如果出现 `CUDA out of memory`，按以下顺序调整：

```bash
# 方案1: 降低 total_tokens (影响最大)
total_tokens=3584  # 从7168降低

# 方案2: 降低 batch_size
batch_per_device=1
global_batch_size=32  # 从64降低

# 方案3: 启用/增加CPU offloading (已在 zero3_l40.json 中启用)
# 在 scripts/zero3_l40.json 中确保:
# "offload_optimizer": {"device": "cpu", ...}
# "offload_param": {"device": "cpu", ...}
```

### 2. 模型加载失败

```bash
# 检查模型路径
ls -la ./model/

# 应该能看到:
# - config.json
# - pytorch_model.bin 或 model.safetensors
# - tokenizer.json 等
```

### 3. DeepSpeed 错误

```bash
# 如果遇到 NCCL 错误，尝试:
export NCCL_P2P_DISABLE=1

# 如果遇到内存不足错误，减少工作进程:
dataloader_num_workers=2  # 从4降低
```

---

## 参数调整总结

根据你的L40数量和显存情况，参考下表：

| 配置 | GPU数 | total_tokens | batch_size | 预估显存 | 适用场景 |
|------|-------|--------------|------------|----------|----------|
| 保守 | 1 | 3584 | 8 | ~20GB | 单卡测试 |
| 标准 | 1 | 7168 | 16 | ~35GB | 单卡完整训练 |
| 多卡 | 4 | 7168 | 64 | ~40GB/卡 | 4卡完整训练 |
| 高性能 | 8 | 7168 | 128 | ~40GB/卡 | 8卡大规模训练 |

---

## 联系与支持

如果遇到问题：
1. 检查本指南的故障排查部分
2. 查看原始README.md获取详细说明
3. 检查GPU显存使用情况: `watch -n 1 nvidia-smi`
