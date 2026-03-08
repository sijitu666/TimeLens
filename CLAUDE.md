# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TimeLens is a video temporal grounding project that trains Multimodal LLMs (based on Qwen2.5-VL and Qwen3-VL) to perform video temporal grounding tasks. The project includes:

- **TimeLens-7B**: Fine-tuned from Qwen2.5-VL-7B-Instruct
- **TimeLens-8B**: Fine-tuned from Qwen3-VL-8B-Instruct
- **TimeLens-Bench**: High-quality evaluation benchmark
- **TimeLens-100K**: Large-scale training dataset

## Environment Setup

```bash
# Create conda environment
conda create -n timelens python=3.11 -y
conda activate timelens

# Install base dependencies
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124

# Install training dependencies (optional)
pip install -r requirements_train.txt

# Install flash-attn (required for both training and inference)
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

## Common Development Commands

### Evaluation

```bash
# Evaluate TimeLens-8B on all datasets (default)
model_path="TencentARC/TimeLens-8B" bash scripts/eval_timelens_bench.sh

# Evaluate specific model on specific datasets with specific GPUs
CUDA_VISIBLE_DEVICES=0,1 \
datasets="activitynet-timelens,qvhighlights-timelens" \
model_path="TencentARC/TimeLens-7B" \
bash scripts/eval_timelens_bench.sh

# Compute metrics from existing predictions
python evaluation/compute_metrics.py -f /path/to/results.jsonl
```

### Training

The training pipeline is a 3-stage process for TimeLens-8B:

**Stage 1: SFT on TimeLens-100K (30K sampled)**
```bash
bash train_scripts/run_sft_qwen3_8b.sh --model_path "/path/to/Qwen3-VL-8B-Instruct"
```

**Stage 2: Run filtering inference on full TimeLens-100K**
```bash
bash scripts/filter_data/filter_data_qwen3_vl.sh \
  --model_path "output/TimeLens-8B/sft/<your_sft_run_dir>" \
  --dataset gemini_refined_data
```

**Stage 3: GRPO training from SFT checkpoint**
```bash
# Training + evaluation
bash train_scripts/run_grpo_and_eval_qwen3_8b.sh \
  --model_path "output/TimeLens-8B/sft/<your_sft_run_dir>" \
  --raw_anno_path "output/TimeLens-8B/filter-data/<your_filter_run_dir>/gemini_refined_data.jsonl"

# Training only
bash train_scripts/run_grpo_qwen3_8b.sh \
  --model_path "output/TimeLens-8B/sft/<your_sft_run_dir>" \
  --raw_anno_path "output/TimeLens-8B/filter-data/<your_filter_run_dir>/gemini_refined_data.jsonl"
```

## Code Architecture

### Directory Structure

```
TimeLens/
├── timelens/                   # Core library code
│   ├── dataset/                # Dataset loading utilities
│   │   └── timelens_data.py    # Dataset classes for TimeLens-Bench and TimeLens-100K
│   └── utils.py                # Utility functions (IoU, timestamp extraction, etc.)
├── training/                   # Training code
│   ├── train/                  # Training scripts
│   │   ├── train_sft_timelens.py   # SFT training
│   │   ├── train_grpo_timelens.py  # GRPO training
│   │   └── train_utils.py
│   ├── trainer/                # Custom trainers
│   │   ├── sft_trainer.py
│   │   └── grpo_trainer_qwenvl.py
│   ├── data/                   # Data loading and preprocessing
│   │   ├── grounding.py
│   │   ├── hybrid.py
│   │   ├── collator.py
│   │   └── preprocess.py
│   ├── filter/                 # Filtering inference for GRPO
│   │   └── infer_qwen3_vl_tvg_dataloader_filter_data.py
│   ├── model_loader.py         # Model loading utilities
│   └── params.py               # Argument dataclasses
├── evaluation/                 # Evaluation code
│   ├── eval_dataloader.py      # Multi-GPU inference with DataLoader
│   ├── compute_metrics.py      # Metric computation (IoU, Recall)
│   └── utils.py                # Evaluation utilities
├── scripts/                    # Shell scripts
│   ├── eval_timelens_bench.sh  # Main evaluation script
│   ├── filter_data/            # Data filtering scripts
│   ├── zero1.json              # DeepSpeed ZeRO-1 config
│   └── zero3.json              # DeepSpeed ZeRO-3 config
├── train_scripts/            # Training shell scripts
│   ├── run_sft_qwen3_8b.sh     # SFT training script
│   ├── run_grpo_qwen3_8b.sh    # GRPO training script
│   └── run_grpo_and_eval_qwen3_8b.sh
└── data/                       # Data directory (user-created)
    ├── TimeLens-Bench/         # Evaluation datasets
    └── TimeLens-100K/          # Training dataset
```

### Key Components

**Dataset Loading** (`timelens/dataset/timelens_data.py`):
- `ActivitynetTimeLensDataset`, `QVHighlightsTimeLensDataset`, `CharadesTimeLensDataset`: Load evaluation datasets
- `TimeLens100KDataset`: Load training dataset
- All datasets follow a consistent interface with `load_annos()` class method

**Training Pipeline**:
- SFT training (`training/train/train_sft_timelens.py`): Supervised fine-tuning on TimeLens-100K
- GRPO training (`training/train/train_grpo_timelens.py`): Reinforcement learning with verifiable rewards
- Custom trainers in `training/trainer/` extend HuggingFace trainers with TimeLens-specific functionality

**Evaluation** (`evaluation/eval_dataloader.py`):
- Multi-GPU inference using DataLoader with multiple workers
- Supports model sharding via `device_map="auto"`
- Chunk-based evaluation for distributed inference across GPUs

**Timestamp Extraction** (`timelens/utils.py`):
- `extract_time()`: Extracts timestamps from model output text
- Supports multiple formats: HH:MM:SS, MM:SS, "X to Y", "X - Y"
- `iou()`: Computes Intersection over Union for temporal spans

## Model Configuration

### Key Hyperparameters

**Video Processing**:
- `fps`: Frames per second for video sampling (default: 2)
- `min_tokens`: Minimum tokens for video encoding (default: 64)
- `total_tokens`: Total tokens for video encoding (default: 14336)
- `fps_max_frames`: Maximum frames per video (computed from tokens)

**Training**:
- `global_batch_size`: Global batch size (SFT: 128, GRPO: 64)
- `learning_rate`: 1e-5 for SFT, 1e-6 for GRPO
- `target_size`: Number of samples per epoch (SFT: 30000, GRPO: 2500)

**GRPO-Specific**:
- `num_generations`: Number of generations per prompt (default: 8)
- `temperature`: Sampling temperature (default: 1.0)
- `reward_funcs`: Reward function (default: "tiou" - temporal IoU)

### Supported Models

- `TencentARC/TimeLens-7B` (based on Qwen2.5-VL-7B-Instruct)
- `TencentARC/TimeLens-8B` (based on Qwen3-VL-8B-Instruct)
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`

## Important Notes

**Flash Attention**: Required for both training and inference. Install with:
```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

**DeepSpeed**: Training uses DeepSpeed for distributed training. Configs are in `scripts/zero1.json` (SFT) and `scripts/zero3.json` (GRPO).

**Data Format**: The training and evaluation code expects specific JSON/JSONL formats. Refer to `timelens/dataset/timelens_data.py` for the expected format.

**Multi-GPU Evaluation**: The evaluation script automatically uses all available GPUs. Set `CUDA_VISIBLE_DEVICES` to control which GPUs to use.

**Output Structure**: Training outputs are saved to:
- SFT: `output/TimeLens-8B/sft/<run_name>/`
- Filter: `output/TimeLens-8B/filter-data/<run_name>/`
- GRPO: `output/TimeLens-8B/grpo/<run_name>/`
