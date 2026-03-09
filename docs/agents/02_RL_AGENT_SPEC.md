# RL Agent (强化学习Agent) 详细需求规范

## Agent信息

- **Agent名称**: RLAgent
- **代号**: RL
- **职责**: RLVR训练管道升级 (Phase 2)
- **协作Agent**:
  - 上游: ArchitectureAgent (接收时间嵌入和适配器接口)
  - 下游: EvaluationAgent (提供训练好的模型用于评估)
  - 协作: DataProcessingAgent (数据采样和预处理)
- **工期**: 5周 (Week 4-8)

---

## 工作空间

### 负责目录

```
timelens-advanced/
├── training/
│   ├── rl/                        # [RL负责] 强化学习模块
│   │   ├── __init__.py
│   │   ├── reward_models/         # 奖励模型
│   │   │   ├── __init__.py
│   │   │   ├── semantic_alignment.py          # RL-001
│   │   │   ├── clip_wrapper.py                # RL-001
│   │   │   └── composite_reward.py            # RL-002
│   │   │
│   │   ├── difficulty_estimation/ # 难度估计
│   │   │   ├── __init__.py
│   │   │   ├── online_estimator.py            # RL-003
│   │   │   ├── reward_ema_tracker.py          # RL-003
│   │   │   └── difficulty_matrix.py           # RL-004
│   │   │
│   │   ├── sampling/              # 采样策略
│   │   │   ├── __init__.py
│   │   │   ├── stratified_sampler.py          # RL-004
│   │   │   └── gaussian_difficulty_sampler.py # RL-004
│   │   │
│   │   └── grpo_grpo_trainer.py # GRPO训练器增强
│   │
│   ├── train/                     # [RL修改] 训练脚本
│   │   ├── train_grpo_timelens.py           # 集成RL模块
│   │   └── grpo_utils.py                    # GRPO工具函数
│   │
│   └── data/                      # [RL修改] 数据加载
│       └── hybrid.py                          # 支持难度采样
│
├── tests/                         # [RL负责] 测试
│   ├── test_semantic_reward.py
│   ├── test_composite_reward.py
│   ├── test_online_difficulty.py
│   ├── test_difficulty_matrix.py
│   └── test_grpo_integration.py
│
└── docs/agents/rl/                # [RL负责] 文档
    ├── designs/
    │   ├── RL-001-semantic-reward-spec.md
    │   ├── RL-002-composite-reward-spec.md
    │   ├── RL-003-online-difficulty-spec.md
    │   └── RL-004-difficulty-matrix-spec.md
    ├── api/
    │   └── rlvr-pipeline-interface.md
    └── tutorials/
        └── grpo_training_guide.md
```

### 接口文件

**必须实现的接口** (位于 `docs/agents/interfaces/`):

1. `interface_rl.py` - RLAgent对外接口定义
2. `schema_rl.json` - 数据格式Schema
3. `README_RL.md` - 接口使用说明

---

## 任务清单

### Task 2.1: 多维组合奖励函数

#### 2.1.1 视觉-文本对齐模型集成 (RL-001)

```yaml
TaskID: RL-001
Title: 集成CLIP/SigLIP视觉-文本对齐模型
Agent: RLAgent
Priority: P0
EstimatedTime: 5天
Input:
  - Phase 1重构后的VTG模型
  - CLIP/SigLIP预训练模型
  - TimeLens-100K训练数据
Output:
  - NewFile: training/rl/reward_models/semantic_alignment.py
  - NewFile: training/rl/reward_models/clip_wrapper.py
  - TestFile: tests/test_semantic_reward.py
  - Doc: docs/agents/rl/designs/RL-001-semantic-reward-spec.md
TechnicalSpecs:
  clip_model: "google/siglip-so400m-patch14-384"
  freeze_weights: true
  device: "cuda:1"  # 可配置到独立GPU
  batch_size: 8
  reward_normalization: "minmax"  # or "z-score"
SuccessCriteria:
  - clip_inference_speed: "> 100 fps (A100)"
  - semantic_correlation: "> 0.7 (vs human judgment)"
  - memory_footprint: "< 4GB"
Dependencies: [ARCH-002, ARCH-003]
```

#### 2.1.2 组合奖励函数实现 (RL-002)

```yaml
TaskID: RL-002
Title: 实现多维组合奖励函数 r_total
Agent: RLAgent
Priority: P0
EstimatedTime: 4天
Input:
  - RL-001实现的语义奖励模块
  - 原始TimeLens IoU奖励实现
  - GRPO训练框架
Output:
  - ModifiedFile: training/rl/reward_models/composite_reward.py
  - NewFile: training/rl/dynamic_weight_scheduler.py
  - TestFile: tests/test_composite_reward.py
  - Doc: docs/agents/rl/designs/RL-002-composite-reward-spec.md
TechnicalSpecs:
  reward_formula: |
    r_total = α(t) · r_iou + β(t) · r_semantic + γ · r_aux

    where:
    - α(t), β(t): dynamic weights based on training progress
    - Stage 1 (t < 0.3T): α=0.3, β=0.7 (semantic focus)
    - Stage 2 (0.3T ≤ t < 0.7T): α=0.5, β=0.5 (balanced)
    - Stage 3 (t ≥ 0.7T): α=0.7, β=0.3 (localization focus)
  reward_clipping: [-10, 10]
  normalization: "z-score with sliding window"
SuccessCriteria:
  - reward_computation_latency: "< 50ms per sample"
  - reward_variance: "< 120% of baseline IoU reward"
  - ablation_improvement: "> 10% R1@0.5 (α=1,β=0 vs α=0,β=1)"
  - correct_stage_transition: "verified by logging"
Dependencies: [RL-001]
```

### Task 2.2: 在线动态难度评估

#### 2.2.1 在线评估机制实现 (RL-003)

```yaml
TaskID: RL-003
Title: 实现无需离线推理的在线难度评估
Agent: RLAgent
Priority: P0
EstimatedTime: 5天
Input:
  - TimeLens原始离线过滤脚本
  - GRPO训练框架
  - 需要消除的离线推理步骤
Output:
  - ModifiedFile: training/train/train_grpo_timelens.py (在线集成)
  - NewFile: training/rl/online_difficulty_estimator.py
  - NewFile: training/rl/reward_ema_tracker.py
  - TestFile: tests/test_online_difficulty.py
  - Doc: docs/agents/rl/designs/RL-003-online-difficulty-spec.md
TechnicalSpecs:
  ema_update_formula: |
    EMA_t = λ · r_t + (1-λ) · EMA_{t-1}
    where λ ∈ [0.1, 0.3] (configurable)
  difficulty_formula: |
    difficulty = 1 / (EMA + ε)
    where ε = 1e-6 (prevent division by zero)
  normalization: "min-max to [0, 1]"
  buffer_management:
    type: "circular_buffer"
    max_size: 100000  # configurable
    eviction_policy: "LRU"
  memory_optimization:
    hash_table: "sample_id -> EMA"
    periodic_cleanup: "every 1000 steps"
SuccessCriteria:
  - ema_update_latency: "< 10ms per sample"
  - memory_footprint: "< 2GB for 100K samples"
  - ema_difference_vs_offline: "< 1e-3"
  - training_speedup: "> 25% (vs offline)"
  - correct_ema_update: "verified for repeated samples"
Dependencies: []
```

### Task 2.3: 分层难度感知采样

#### 2.3.1 多因子难度评估矩阵 (RL-004)

```yaml
TaskID: RL-004
Title: 实现多维因子加权难度评估矩阵
Agent: RLAgent
Priority: P1
EstimatedTime: 6天
Input:
  - RL-003的在线难度估计框架
  - 空间复杂度评估需求 (目标检测)
  - 时序动态复杂度需求 (光流)
  - 语义复杂度需求
Output:
  - NewFile: training/rl/complexity_analyzer.py
  - NewFile: training/rl/difficulty_matrix.py
  - ModifiedFile: training/rl/online_difficulty_estimator.py (集成)
  - TestFile: tests/test_difficulty_matrix.py
  - Doc: docs/agents/rl/designs/RL-004-difficulty-matrix-spec.md
TechnicalSpecs:
  complexity_dimensions:
    spatial:
      method: "YOLO-nano or DETR-tiny"
      metrics: ["num_objects", "area_variance"]
      output: "spatial_score ∈ [0, 1]"
    temporal_dynamics:
      method: "FlowFormer-tiny or RAFT-lite"
      metrics: ["flow_magnitude_variance", "direction_change_rate"]
      output: "temporal_score ∈ [0, 1]"
    semantic:
      method: "language_model_complexity"
      metrics: ["query_length", "lexical_rarity", "syntactic_complexity"]
      output: "semantic_score ∈ [0, 1]"
  difficulty_matrix:
    type: "3xN matrix"
    dimensions: ["spatial", "temporal", "semantic"]
    normalization: "min-max to [0, 1] per dimension"
    update_mode: "dynamic_online_learning"
  composite_difficulty_formula: |
    difficulty = w₁·spatial + w₂·temporal + w₃·semantic + w₄·historical
    where:
    - w₁, w₂, w₃: dimension weights (learnable or preset)
    - w₄: historical reward weight (from RL-003)
    - Σwᵢ = 1 (normalization)
  gaussian_sampling_fusion:
    base: "original TimeLens Gaussian sampling"
    adjustment: |
      μ_new = μ + α·difficulty
      σ_new = σ·(1 - β·difficulty)
    where α, β are configurable hyperparameters
SuccessCriteria:
  - spatial_complexity_latency: "< 50ms per frame"
  - optical_flow_latency: "< 200ms per frame (GPU)"
  - difficulty_correlation: "> 0.6 (vs human annotation)"
  - sampling_distribution: "matches expected Gaussian adjustment"
  - extra_computation_overhead: "< 20% (vs baseline)"
Dependencies: [RL-003]
```

---

## 接口契约

### 对外接口 (`docs/agents/interfaces/interface_rl.py`)

```python
from typing import Protocol, Dict, List, Any, Optional
from torch import Tensor

class RLVRPipelineInterface(Protocol):
    """RLVR训练管道接口 - RLAgent核心对外接口"""

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        执行一步GRPO训练

        Args:
            batch: 包含以下键的字典
                - "video_features": [B, T, D]
                - "queries": List[str]
                - "ground_truth_spans": List[Tuple[float, float]]
                - "sample_ids": List[str]

        Returns:
            metrics: 包含以下键的字典
                - "loss": float
                - "reward": float
                - "kl_div": float
                - "advantage": float
        """
        ...

    def get_sample_weights(self, sample_ids: List[str]) -> Tensor:
        """
        获取样本的当前难度权重

        Args:
            sample_ids: 样本ID列表

        Returns:
            weights: [len(sample_ids)] 权重张量
        """
        ...

    def compute_reward(
        self,
        predicted_spans: List[Tuple[float, float]],
        ground_truth_spans: List[Tuple[float, float]],
        video_features: Tensor,
        queries: List[str]
    ) -> Dict[str, float]:
        """
        计算组合奖励

        Args:
            predicted_spans: 预测的时间段列表
            ground_truth_spans: 真实时间段列表
            video_features: 视频特征 [B, T, D]
            queries: 文本查询列表

        Returns:
            rewards: 包含以下键的字典
                - "total": float (r_total)
                - "iou": float (r_iou)
                - "semantic": float (r_semantic)
                - "auxiliary": float (r_aux)
        """
        ...

    def export_checkpoint(self, path: str, include_optimizer: bool = False) -> None:
        """
        导出训练状态检查点

        Args:
            path: 保存路径
            include_optimizer: 是否包含优化器状态
        """
        ...

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        加载训练状态检查点

        Args:
            path: 检查点路径

        Returns:
            checkpoint_data: 包含训练状态的字典
        """
        ...


class DifficultyEstimationInterface(Protocol):
    """难度估计接口 - 供其他Agent查询"""

    def update_ema(self, sample_id: str, reward: float) -> None:
        """更新样本的EMA"""
        ...

    def get_difficulty(self, sample_id: str) -> float:
        """获取样本难度分数"""
        ...

    def get_difficulty_matrix(self, sample_ids: List[str]) -> Tensor:
        """获取多维难度矩阵"""
        ...


class RewardFunctionInterface(Protocol):
    """奖励函数接口 - 用于组件化奖励计算"""

    def compute_iou_reward(
        self,
        predicted: Tuple[float, float],
        ground_truth: Tuple[float, float]
    ) -> float:
        """计算IoU奖励"""
        ...

    def compute_semantic_reward(
        self,
        video_features: Tensor,
        query: str
    ) -> float:
        """计算语义奖励"""
        ...
```

---

## 依赖声明

```yaml
agent: RLAgent
dependencies:
  upstream:
    - agent: ArchitectureAgent
      interface: temporal_embedding_interface
      version: ">=1.0,<2.0"
      required_tasks: [ARCH-002, ARCH-003]

    - agent: DataProcessingAgent
      interface: dataset_sampler_interface
      version: ">=1.0,<2.0"

  downstream:
    - agent: EvaluationAgent
      interface: rlvr_pipeline_interface
      version: ">=1.0,<2.0"
      provided_tasks: [RL-001, RL-002, RL-003, RL-004]
```

---

## 验收标准

### 功能验收

- [ ] 语义奖励计算延迟 < 50ms/样本
- [ ] 组合奖励方差 < 原始IoU奖励的120%
- [ ] EMA更新延迟 < 10ms/样本
- [ ] 在线训练速度提升 > 25% (相比离线)
- [ ] 难度矩阵计算延迟 < 200ms/批次

### 集成验收

- [ ] 与ArchitectureAgent接口兼容
- [ ] 与EvaluationAgent接口兼容
- [ ] 所有单元测试通过率 > 95%
- [ ] 集成测试通过率 100%
- [ ] 代码覆盖率 > 80%

### 性能验收

- [ ] GRPO训练步时间 < 2s/步 (4x L40)
- [ ] 显存占用 < 42GB/卡 (L40)
- [ ] 支持至少2路并发训练
- [ ] 检查点保存时间 < 30s

---

**文档结束**

*本文档由RLAgent维护，版本变更请提交PR*
