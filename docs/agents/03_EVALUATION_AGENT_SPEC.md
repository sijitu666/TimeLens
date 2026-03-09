# Evaluation Agent (测试验证Agent) 详细需求规范

## Agent信息

- **Agent名称**: EvaluationAgent
- **代号**: EVAL
- **职责**: 评估与对齐 (Phase 3) - 基线对比、消融实验、性能Profiling
- **协作Agent**:
  - 上游: ArchitectureAgent, RLAgent (接收训练好的模型和组件)
  - 下游: ProjectManager (提供最终评估报告)
  - 协作: DataProcessingAgent (评估数据准备)
- **工期**: 3周 (Week 8-10)

---

## 工作空间

### 负责目录

```
timelens-advanced/
├── evaluation/                    # [EVAL负责] 评估模块
│   ├── __init__.py
│   ├── benchmark/                 # 基准测试
│   │   ├── __init__.py
│   │   ├── timelens_bench.py                  # TimeLens-Bench评估
│   │   ├── high_freq_benchmark.py             # 高频动作基准
│   │   └── long_video_benchmark.py            # 长视频基准
│   │
│   ├── metrics/                   # 评估指标
│   │   ├── __init__.py
│   │   ├── temporal_iou.py                    # 时间IoU
│   │   ├── recall_at_k.py                     # R1@k指标
│   │   ├── miou.py                            # mIoU
│   │   └── semantic_alignment.py              # 语义对齐度
│   │
│   ├── ablation/                  # 消融实验
│   │   ├── __init__.py
│   │   ├── ablation_runner.py                 # 消融实验运行器
│   │   ├── continuous_embedding_ablation.py   # 连续嵌入消融
│   │   ├── temporal_adapter_ablation.py     # 时序适配器消融
│   │   ├── reward_function_ablation.py      # 奖励函数消融
│   │   └── online_difficulty_ablation.py      # 在线难度消融
│   │
│   └── profiling/                 # 性能分析
│       ├── __init__.py
│       ├── latency_profiler.py                # 延迟分析
│       ├── memory_profiler.py                 # 显存分析
│       ├── throughput_profiler.py             # 吞吐量分析
│       └── visualization.py                   # 可视化工具
│
├── tests/evaluation/              # [EVAL负责] 评估测试
│   ├── test_metrics.py
│   ├── test_benchmark.py
│   └── test_ablation.py
│
├── scripts/evaluation/            # [EVAL负责] 评估脚本
│   ├── run_full_evaluation.sh
│   ├── run_ablation_study.sh
│   ├── run_benchmark.sh
│   └── generate_report.py
│
├── reports/                       # [EVAL输出] 评估报告
│   ├── baseline_comparison/
│   │   ├── report.pdf
│   │   ├── metrics.json
│   │   └── plots/
│   ├── ablation_study/
│   │   ├── report.pdf
│   │   ├── results.json
│   │   └── learning_curves/
│   └── profiling/
│       ├── latency_report.pdf
│       ├── memory_timeline.json
│       └── flame_graphs/
│
└── docs/agents/evaluation/        # [EVAL负责] 评估文档
    ├── designs/
    │   ├── EVAL-001-baseline-comparison-spec.md
    │   ├── EVAL-002-profiling-spec.md
    │   └── EVAL-003-reporting-spec.md
    ├── api/
    │   └── evaluation-interface.md
    └── tutorials/
        └── running_evaluation.md
```

### 接口文件

**必须实现的接口** (位于 `docs/agents/interfaces/`):

1. `interface_evaluation.py` - EvaluationAgent对外接口
2. `schema_evaluation.json` - 数据格式Schema
3. `README_EVALUATION.md` - 接口使用说明

---

## 任务清单

### Task 3.1: 基线对比与消融实验 (EVAL-001)

```yaml
TaskID: EVAL-001
Title: TimeLens-Bench基线对比与消融实验
Agent: EvaluationAgent
Priority: P0
EstimatedTime: 7天
Input:
  - 原始TimeLens-7B/8B模型检查点
  - Phase 1重构后的架构 (连续时间嵌入 + 时序适配器)
  - Phase 2升级后的RLVR管道 (多维奖励 + 在线难度)
  - TimeLens-Bench评估数据集
Output:
  - Report: reports/baseline_comparison/report.pdf
  - Data: reports/baseline_comparison/metrics.json
  - Plots: reports/baseline_comparison/plots/
  - Code: evaluation/benchmark/timelens_bench.py
  - Doc: docs/agents/evaluation/designs/EVAL-001-baseline-comparison-spec.md
ExperimentalDesign:
  baseline_comparison:
    models:
      - name: "TimeLens-8B-Original"
        checkpoint: "TencentARC/TimeLens-8B"
        description: "原始基线模型"
      - name: "TimeLens-8B-Advanced"
        checkpoint: "output/timelens-advanced/final.ckpt"
        description: "本项目的完整优化模型"
    datasets:
      - "charades-timelens"
      - "activitynet-timelens"
      - "qvhighlights-timelens"
    metrics:
      - "R1@0.3"
      - "R1@0.5"
      - "R1@0.7"
      - "mIoU"
    statistics:
      - "mean over 3 random seeds"
      - "standard deviation"
      - "statistical significance test (p < 0.05)"
  ablation_studies:
    continuous_embedding_ablation:
      variants:
        - name: "full_system"
          config: "all components enabled"
        - name: "w/o_continuous_embedding"
          config: "use original text timestamp"
      focus_metric: "high-frequency action localization accuracy"
    temporal_adapter_ablation:
      variants:
        - name: "full_system"
          config: "with temporal adapter"
        - name: "w/o_temporal_adapter"
          config: "direct visual to LLM"
      focus_metric: "long video modeling capability"
    reward_function_ablation:
      variants:
        - name: "α=1.0, β=0.0"
          config: "IoU only"
        - name: "α=0.5, β=0.5"
          config: "balanced"
        - name: "α=0.0, β=1.0"
          config: "semantic only"
      focus_metric: "convergence speed and final performance"
    online_difficulty_ablation:
      variants:
        - name: "online_ema"
          config: "online EMA update"
        - name: "offline_precomputed"
          config: "original TimeLens offline"
      focus_metric: "training time and stability"
AcceptanceCriteria:
  - baseline_r1_at_0.5_improvement: "> 3% vs original TimeLens"
  - continuous_embedding_high_freq_improvement: "> 8% on >30fps"
  - temporal_adapter_long_video_improvement: "> 5% mIoU on >5min"
  - reward_ablation_gap: "> 10% between best and worst"
  - online_difficulty_speedup: "> 25% training time reduction"
  - statistical_significance: "p < 0.05 for all claims"
Dependencies: [ARCH-003, ARCH-006, RL-004]
```

### Task 3.2: 性能与显存Profiling (EVAL-002)

```yaml
TaskID: EVAL-002
Title: 性能Profiling与显存优化验证
Agent: EvaluationAgent
Priority: P1
EstimatedTime: 5天
Input:
  - Phase 1和2的所有优化组件
  - Profiling工具 (PyTorch Profiler, Nsight Systems)
  - 目标部署硬件规格 (L40 48GB x4)
Output:
  - Report: reports/profiling/latency_report.pdf
  - Data: reports/profiling/memory_timeline.json
  - Visualization: reports/profiling/flame_graphs/
  - Recommendations: docs/agents/evaluation/EVAL-002-optimization-suggestions.md
  - Code: evaluation/profiling/
ProfilingPlan:
  latency_profiling:
    end_to_end:
      video_lengths: ["30s", "60s", "300s"]
      metrics: ["P50", "P95", "P99"]
      measurement: "full pipeline: preprocessing -> encoding -> adapter -> LLM -> output"
      target: "P95 < 2x video duration"
    module_breakdown:
      modules:
        - name: "Visual Encoder"
          description: "Qwen3 ViT"
        - name: "Temporal Adapter"
          description: "Mamba blocks"
        - name: "LLM"
          description: "Qwen3 Decoder"
        - name: "Post-processing"
          description: "timestamp extraction"
      output: "latency breakdown by module, identify bottleneck"
  memory_profiling:
    peak_analysis:
      measurement: "full training workflow memory timeline"
      breakdown:
        - category: "model_parameters"
          description: "weights and biases"
        - category: "activations"
          description: "forward/backward activations"
        - category: "optimizer_states"
          description: "Adam moments etc."
        - category: "temporary_buffers"
          description: "intermediate results"
      peak_identification: "identify peak memory points (usually forward/backward)"
    component_footprint:
      components:
        - name: "Qwen3 ViT"
          target: "< X GB"
        - name: "Temporal Adapter"
          target: "< 2GB"
        - name: "Qwen3 LLM"
          target: "< Z GB"
      validation: "ensure peak < 40GB on L40 (leave 8GB margin)"
  training_speed:
    throughput:
      metric: "samples per second"
      comparison: "online difficulty estimation vs offline"
      target: "online > 25% speedup"
  concurrency:
    multi_way:
      setup: "4x L40"
      scenarios: ["2-way", "4-way" concurrent]
      metrics: ["latency", "throughput"]
      target: "2-way latency increase < 50%"
AcceptanceCriteria:
  - end_to_end_p95_latency: "< 2x video duration (all lengths)"
  - peak_memory: "< 40GB (L40 48GB, 8GB margin)"
  - temporal_adapter_memory: "< 2GB"
  - online_vs_offline_speedup: "> 25%"
  - two_way_latency_increase: "< 50%"
  - complete_profiling_report: "bottleneck analysis and optimization suggestions"
Dependencies: [ARCH-006, RL-004]
```

---

## 工作流与协作

### 开发周期

ArchitectureAgent遵循 **1周迭代周期**：

| 日期 | 活动 | 说明 |
|-----|------|------|
| 周一 | 周会同步 | 与所有Agent同步进度，调整计划 |
| 周二-周四 | 独立开发 | 专注完成分配的任务 |
| 周五 | 代码审查 | 提交PR，Peer Review |
| 周六 | 集成测试 | 合并到develop分支，运行测试 |
| 周日 | 文档更新 | 更新技术文档，准备下周计划 |

### 关键协作点

1. **向RLAgent交付** (Week 4周末):
   - 接口: `TemporalEmbeddingInterface`
   - 组件: `ContinuousTemporalEmbedding`, `TemporalAdapter`
   - 文档: 接口使用说明、集成指南

2. **向EvaluationAgent交付** (Week 8周末):
   - 组件: 完整优化后的模型
   - 接口: 模型评估接口、导出功能
   - 文档: 性能基准、已知问题

### 代码提交规范

```bash
# 功能开发
git checkout -b feature/arch-002-continuous-embedding
# ... 开发 ...
git commit -m "feat(arch): implement continuous temporal embedding layer

- Add MLP-based time encoder with microsecond precision
- Implement gradient checkpointing for memory efficiency
- Include comprehensive unit tests

Refs: ARCH-002"

# Bug修复
git checkout -b bugfix/arch-001-dataflow
# ... 修复 ...
git commit -m "fix(arch): correct timestamp data flow in preprocessing

- Remove text tokenization of timestamps
- Preserve float precision through pipeline
- Fix edge case handling for sub-second precision

Fixes: ARCH-001"

# 文档更新
git commit -m "docs(arch): add architecture diagram and API docs

- Include system architecture overview
- Document public API interfaces
- Add usage examples and best practices

Refs: ARCH-DOCS"
```

---

## 附录

### A. 工具链

- **代码开发**: VSCode / PyCharm
- **版本控制**: Git + GitHub/GitLab
- **项目管理**: Notion / Linear / Jira
- **文档**: Markdown + MkDocs
- **绘图**: draw.io / Excalidraw

### B. 参考资源

- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [ONNX Export Guide](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [TimeLens Paper](https://arxiv.org/abs/2512.14698)

---

**文档结束**

*本文档由EvaluationAgent维护，版本变更请提交PR*
