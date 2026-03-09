# 视频时间定位 (VTG) 模型进阶优化 - 多Agent协作开发规范

## 项目概述

**项目代号**: VTG-Advanced-2025
**基础架构**: TimeLens (Qwen3-VL-8B)
**目标场景**: 高帧率细粒度动作定位 (机器人操作误差分析) + 长视频MLOps部署
**总工期**: 12周
**开发模式**: Multi-Agent协作开发

---

## Agent组织架构

```
┌─────────────────────────────────────────────────────────────┐
│                    项目管理 Agent                            │
│         (Project Manager - 进度协调、资源分配)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼───────┐ ┌──────▼──────┐ ┌─────▼────────┐
│  架构设计      │ │  强化学习   │ │  测试验证   │
│  Agent        │ │  Agent      │ │  Agent      │
│  (Phase 1)    │ │  (Phase 2)  │ │  (Phase 3)  │
└─────┬─────────┘ └─────┬───────┘ └──────┬──────┘
      │                 │                │
      │   ┌─────────────┴──────────────┐ │
      │   │     数据处理 Agent         │ │
      └──►│  (Dataset Pipeline)       │◄┘
          └───────────────────────────┘
```

---

## Phase 1: 架构与时间编码重构

**负责 Agent**: ArchitectureAgent
**协作 Agent**: DataProcessingAgent (数据预处理改造)
**工期**: 4周
**输入**: TimeLens原始代码库
**输出**: 重构后的时间编码模块 + 时序适配器

### Task 1.1: 连续时间嵌入开发

**子任务 1.1.1: 时间Token离散化逻辑移除**

```yaml
TaskID: ARCH-001
Title: 移除文本时间戳预处理逻辑
Agent: ArchitectureAgent
Priority: P0
EstimatedTime: 3天

Input:
  - File: training/data/grounding.py
  - File: training/data/preprocess.py
  - TimeLens原始时间格式化逻辑

Output:
  - ModifiedFile: training/data/grounding.py (移除text timestamp转换)
  - ModifiedFile: training/data/preprocess.py (移除timestamp tokenization)
  - Doc: ARCH-001-removal-spec.md (移除逻辑文档)

AcceptanceCriteria:
  1. 原代码中的timestamp转text token逻辑被完全移除
  2. 保留原始浮点时间戳数据格式
  3. 通过单元测试验证数据流正确性
  4. 不产生除零或越界错误

Dependencies: []
```

**子任务 1.1.2: 连续时间嵌入层设计**

```yaml
TaskID: ARCH-002
Title: 设计并实现连续时间嵌入层
Agent: ArchitectureAgent
Priority: P0
EstimatedTime: 7天

Input:
  - Qwen3-VL-8B模型架构
  - TimeLens现有模型集成点
  - Task ARCH-001的输出(浮点时间戳格式)

Output:
  - NewFile: training/models/continuous_temporal_embedding.py
  - NewFile: training/models/temporal_rope.py (1D-RoPE实现)
  - ModifiedFile: training/train/train_sft_timelens.py (集成点)
  - TestFile: tests/test_continuous_embedding.py
  - Doc: ARCH-002-embedding-design.md

TechnicalSpecs:
  1. 时间嵌入维度: 与Qwen3 hidden size一致 (3584)
  2. 时间范围: 支持 0.001s ~ 10000s (微秒级精度)
  3. 编码方式:
     - 方案A: 可学习的MLP (输入: 标量时间 -> 输出: 时间嵌入向量)
     - 方案B: 1D-RoPE旋转位置编码 (针对时间轴)
  4. 集成点: 在Qwen3的visual merger之后，LLM之前

AcceptanceCriteria:
  1. 时间嵌入层能正确处理浮点时间输入
  2. 输出维度与模型hidden size匹配
  3. 支持微秒级精度的时间差计算
  4. 通过梯度检查 (梯度不消失/爆炸)
  5. 在合成数据上展示时间顺序感知能力

Dependencies: [ARCH-001]
```

**子任务 1.1.3: 高频动作捕捉优化**

```yaml
TaskID: ARCH-003
Title: 高频细粒度动作时间建模优化
Agent: ArchitectureAgent
Priority: P1
EstimatedTime: 4天

Input:
  - ARCH-002实现的连续时间嵌入
  - 高帧率视频数据 (60fps+)
  - 机器人操作误差分析场景需求

Output:
  - ModifiedFile: training/models/continuous_temporal_embedding.py (高频优化)
  - NewFile: training/models/high_freq_temporal_attention.py
  - TestFile: tests/test_high_freq_capture.py
  - Doc: ARCH-003-high-freq-optimization.md

TechnicalSpecs:
  1. 高频时间分解:
     - 粗粒度: 秒级 (由ARCH-002处理)
     - 细粒度: 毫秒/微秒级 (本任务)
  2. 实现双尺度时间嵌入:
     - coarse_embed = Embed(floor(time))
     - fine_embed = Embed(frac(time) * 1000)  # 毫秒部分
     - final_embed = fusion(coarse_embed, fine_embed)
  3. 针对机器人操作误差场景:
     - 支持0.01s (10ms) 精度的事件边界检测

AcceptanceCriteria:
  1. 能正确区分相差 <0.1s 的时间点
  2. 在模拟高频数据集上IoU比基线提升>5%
  3. 计算 overhead < 15%

Dependencies: [ARCH-002]
```

### Task 1.2: 轻量级时序适配器研发

**子任务 1.2.1: 时序状态传递机制设计**

```yaml
TaskID: ARCH-004
Title: 设计时序状态传递机制 (消除显式时间Token)
Agent: ArchitectureAgent
Priority: P0
EstimatedTime: 5天

Input:
  - TimeLens原始交错时间前缀方案
  - Qwen3-VL特征提取流程
  - 流式处理需求规格

Output:
  - NewFile: training/models/temporal_adapter.py (核心适配器)
  - NewFile: training/models/temporal_state_bank.py (状态存储)
  - Doc: ARCH-004-temporal-state-design.md
  - Diagram: temporal-adapter-arch.png

TechnicalSpecs:
  1. 架构位置:
     Input Video -> [Qwen3 Visual Encoder] -> [Visual Merger]
     -> [Temporal Adapter] -> [Qwen3 LLM] -> Output

  2. Temporal Adapter 设计:
     - 输入: frame_features [T, D] (T=时间步, D=特征维)
     - 输出: temporal_enhanced_features [T, D]
     - 内部: 轻量级双向LSTM或Mamba Block

  3. 状态传递机制:
     - 显式时间Token -> 隐式状态向量
     - 状态维度: 256-dim (轻量级)
     - 传递方式: cross-attention 到LLM层

AcceptanceCriteria:
  1. 消除所有显式时间Token生成逻辑
  2. 帧间时间关系通过隐式状态传递
  3. 相比原始方案，上下文长度减少 >30%
  4. 在短序列(<2min视频)上保持精度不降

Dependencies: [ARCH-001, ARCH-002]
```

**子任务 1.2.2: Mamba时序适配器实现**

```yaml
TaskID: ARCH-005
Title: 实现基于Mamba的轻量级时序适配器
Agent: ArchitectureAgent
Priority: P1
EstimatedTime: 6天

Input:
  - ARCH-004设计的时序适配器架构
  - Mamba/mamba2模型实现参考
  - ONNX/TensorRT算子兼容性需求

Output:
  - NewFile: training/models/mamba_temporal_adapter.py
  - NewFile: training/models/mamba_block.py (核心block实现)
  - TestFile: tests/test_mamba_adapter.py
  - Doc: ARCH-005-mamba-adapter-impl.md
  - ExportTest: test_onnx_export.py

TechnicalSpecs:
  1. Mamba Block设计 (轻量版):
     - d_model: 512 (vs 标准2048)
     - d_state: 64 (状态空间维度)
     - d_conv: 3 (局部卷积宽度)
     - expand_factor: 2 (扩展因子)

  2. 整体架构:
     Input [T, 3584] (Qwen3 visual dim)
     -> Linear(3584->512)
     -> MambaBlock(x2, residual)
     -> Linear(512->3584)
     -> Output [T, 3584]

  3. ONNX/TensorRT兼容性:
     - 避免动态shape操作
     - 使用固定max_seq_len (如2048)
     - 提供padding/mask处理
     - 测试export: torch.onnx.export()

AcceptanceCriteria:
  1. 参数量 < 10M (轻量级)
  2. 前向推理速度 > 1000 tokens/sec (A100)
  3. 成功导出ONNX格式
  4. 在合成时序数据上验证选择性记忆能力
  5. 相比LSTM，长序列(>1000步)建模能力提升>20%

Dependencies: [ARCH-004]
```

**子任务 1.2.3: 流式推理适配与ONNX导出**

```yaml
TaskID: ARCH-006
Title: 流式推理适配与生产环境部署优化
Agent: ArchitectureAgent
Priority: P1
EstimatedTime: 5天

Input:
  - ARCH-005实现的Mamba时序适配器
  - 流式处理需求规格
  - MLOps部署环境要求

Output:
  - NewFile: inference/streaming_vtg_pipeline.py
  - NewFile: inference/onnx_exporter.py
  - NewFile: inference/tensorrt_optimizer.py
  - Doc: ARCH-006-deployment-guide.md
  - Benchmark: latency_benchmark.json

TechnicalSpecs:
  1. 流式处理架构:
     - 视频分块输入 (chunk_size: 30s)
     - 状态缓存机制 (state_buffer)
     - 滑动窗口推理 (overlap: 5s)
     - 异步I/O处理

  2. ONNX导出优化:
     - 动态轴: batch_size, seq_len
     - 算子融合: LayerNorm, GELU
     - 量化: INT8 (可选)
     - 验证: ONNX Runtime推理一致性

  3. TensorRT优化 (可选):
     - FP16/INT8精度
     - Kernel Auto-Tuning
     - 显存池优化

AcceptanceCriteria:
  1. 流式延迟 < 2x 视频时长 (即2倍速处理)
  2. 首包延迟 < 5s (从视频输入到首结果)
  3. ONNX推理速度与PyTorch差距 < 20%
  4. 显存占用在L40上 < 40GB (留8GB余量)
  5. 支持并发: 至少2路视频同时处理

Dependencies: [ARCH-005]
```

---

## Phase 2: RLVR 训练管道升级

**负责 Agent**: RLAgent
**协作 Agent**: DataProcessingAgent (数据流程改造), ArchitectureAgent (模型接口适配)
**工期**: 5周
**输入**: Phase 1重构后的模型架构
**输出**: 升级后的RLVR训练管道

### Task 2.1: 多维组合奖励函数实现

**子任务 2.1.1: 视觉-文本对齐模型集成**

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
  - Doc: RL-001-semantic-reward-spec.md

TechnicalSpecs:
  1. CLIP/SigLIP集成:
     - 模型: SigLIP-SO400M (推荐，更好的细粒度对齐)
     - 冻结权重 (不训练)
     - 设备: 可配置到独立GPU

  2. 语义相似度计算:
     - 输入: 视频片段帧 [F, 3, H, W] + 文本查询
     - 输出: 余弦相似度分数 [-1, 1]
     - 处理: 帧级特征平均池化

  3. Reward转换:
     - r_semantic = (cosine_sim + 1) / 2  # 归一化到[0,1]

AcceptanceCriteria:
  1. CLIP推理速度 > 100 fps (单张A100)
  2. 语义奖励与人工判断相关性 > 0.7
  3. 显存占用 < 4GB (SigLIP-SO400M)
  4. 支持batch推理 (batch_size >= 8)

Dependencies: [ARCH-002, ARCH-003]
```

**子任务 2.1.2: 组合奖励函数实现**

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
  - ModifiedFile: training/rl/reward_funcs.py (组合奖励)
  - NewFile: training/rl/dynamic_weight_scheduler.py
  - TestFile: tests/test_composite_reward.py
  - Doc: RL-002-composite-reward-spec.md

TechnicalSpecs:
  1. 组合奖励公式:
     r_total = α(t) · r_iou + β(t) · r_semantic + γ · r_aux

     其中:
     - r_iou: 时间IoU奖励 (原始TimeLens)
     - r_semantic: 视觉-文本对齐奖励 (RL-001)
     - r_aux: 辅助奖励 (可选，如长度正则化)
     - α(t), β(t): 动态权重 (见下)

  2. 动态权重退火策略:
     - 阶段1 (t < 0.3T): α=0.3, β=0.7 (重视语义)
     - 阶段2 (0.3T ≤ t < 0.7T): α=0.5, β=0.5 (均衡)
     - 阶段3 (t ≥ 0.7T): α=0.7, β=0.3 (精确定位)
     - T: 总训练步数

  3. 实现细节:
     - 奖励归一化: 使用滑动窗口统计 (z-score)
     - 梯度截断: 奖励值限制在 [-10, 10]
     - 可配置: 通过YAML配置文件调整权重

AcceptanceCriteria:
  1. 奖励计算延迟 < 50ms (单样本)
  2. 组合奖励方差 < 原始IoU奖励的120%
  3. 消融实验: α=1,β=0 vs α=0,β=1 性能差距 > 10% R1@0.5
  4. 动态权重调度器可正确按训练步数切换阶段

Dependencies: [RL-001]
```

### Task 2.2: 在线动态难度评估

**子任务 2.2.1: 在线评估机制实现**

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
  - Doc: RL-003-online-difficulty-spec.md

TechnicalSpecs:
  1. 在线评估机制:
     - 在GRPO训练循环中实时计算每个样本的奖励
     - 使用指数移动平均 (EMA) 跟踪每个样本的历史奖励
     - 公式: EMA_t = λ · r_t + (1-λ) · EMA_{t-1}
     - 典型λ值: 0.1 (慢更新) 到 0.3 (快更新)

  2. 难度估计:
     - 难度分数 = 1 / (EMA + ε)  # 奖励越低，难度越高
     - ε = 1e-6 (防止除零)
     - 归一化到 [0, 1] 范围

  3. 消除离线推理:
     - 原流程: 离线跑推理 → 计算IoU → 保存结果 → 训练时读取
     - 新流程: 训练时实时生成 → 即时计算奖励 → 更新EMA → 调整采样权重

  4. 内存优化:
     - 使用固定大小的环形缓冲区存储EMA
     - 哈希表存储样本ID -> EMA的映射
     - 定期清理长期未访问的条目

AcceptanceCriteria:
  1. 在线EMA更新延迟 < 10ms (单样本)
  2. 内存占用 < 2GB (100K样本)
  3. 与离线计算结果的EMA差异 < 1e-3
  4. 训练速度相比离线方案提升 > 30%
  5. 正确支持样本重复出现的EMA更新

Dependencies: []
```

### Task 2.3: 分层难度感知采样

**子任务 2.3.1: 多因子难度评估矩阵**

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
  - Doc: RL-004-difficulty-matrix-spec.md

TechnicalSpecs:
  1. 多维度复杂度评估:

     a) 空间复杂度 (Spatial Complexity)
        - 输入: 视频帧 [H, W, 3]
        - 方法: 使用轻量级目标检测器 (YOLO-nano或DETR-tiny)
        - 指标: 目标数量 + 目标面积方差
        - 输出: spatial_score ∈ [0, 1]

     b) 时序动态复杂度 (Temporal Dynamics)
        - 输入: 连续视频帧序列
        - 方法: 计算光流 (FlowFormer-tiny或RAFT-lite)
        - 指标: 光流幅度方差 + 运动方向变化率
        - 输出: temporal_score ∈ [0, 1]

     c) 语义复杂度 (Semantic Complexity)
        - 输入: 文本查询
        - 方法: 语言模型复杂度估计
        - 指标: 查询长度 + 词汇稀有度 + 句法复杂度
        - 输出: semantic_score ∈ [0, 1]

  2. 难度评估矩阵:
     - 构建 3xN 矩阵 (3维度 x N样本)
     - 每个样本的各维度分数归一化到 [0,1]
     - 支持动态更新 (在线学习)

  3. 综合难度分数计算:
     difficulty = w₁·spatial + w₂·temporal + w₃·semantic + w₄·historical

     其中:
     - w₁, w₂, w₃: 维度权重 (可学习或预设)
     - w₄: 历史奖励权重 (来自RL-003)
     - 所有权重和为1

  4. 高斯采样融合:
     - 在原始TimeLens高斯采样基础上
     - 用综合难度分数调整采样均值和方差
     - 公式: μ_new = μ + α·difficulty, σ_new = σ·(1 - β·difficulty)

AcceptanceCriteria:
  1. 空间复杂度计算延迟 < 50ms/帧
  2. 光流计算延迟 < 200ms/帧 (使用GPU)
  3. 综合难度分数与人工标注相关性 > 0.6
  4. 采样分布符合预期的高斯调整
  5. 额外计算开销 < 20% (相比原始方案)

Dependencies: [RL-003]
```

---

## Phase 3: 评估与对齐

**负责 Agent**: EvaluationAgent
**协作 Agent**: ArchitectureAgent (消融实验支持), RLAgent (奖励函数验证)
**工期**: 3周
**输入**: Phase 1和2的所有组件
**输出**: 完整的评估报告 + 性能基准

### Task 3.1: 基线对比与消融实验

```yaml
TaskID: EVAL-001
Title: TimeLens-Bench基线对比与消融实验
Agent: EvaluationAgent
Priority: P0
EstimatedTime: 7天

Input:
  - 原始TimeLens-7B/8B模型
  - Phase 1重构后的架构 (连续时间嵌入 + 时序适配器)
  - Phase 2升级后的RLVR管道 (多维奖励 + 在线难度)
  - TimeLens-Bench评估数据集

Output:
  - Report: EVAL-001-baseline-comparison.pdf
  - Data: eval_results/baseline_vs_ours.json
  - Plots: ablation_study_plots/ (学习曲线、指标对比)
  - Code: evaluation/benchmark_suite.py

ExperimentalDesign:
  1. 基线对比 (vs 原始TimeLens):
     - 模型: TimeLens-8B (原始) vs TimeLens-8B-Advanced (本项目)
     - 数据集: TimeLens-Bench全部3个子集
     - 指标: R1@0.3, R1@0.5, R1@0.7, mIoU
     - 统计: 3次随机种子平均，报告标准差

  2. 消融实验设计:

     a) 连续时间嵌入消融:
        - 完整系统
        - 完整系统 - 连续嵌入 (使用原始文本时间戳)
        - 对比指标: 高频动作定位精度

     b) 时序适配器消融:
        - 完整系统
        - 完整系统 - 时序适配器
        - 对比指标: 长视频建模能力

     c) 多维奖励消融:
        - 完整系统 (α=0.5, β=0.5)
        - 仅IoU奖励 (α=1.0, β=0)
        - 仅语义奖励 (α=0, β=1.0)
        - 对比指标: 训练收敛速度、最终性能

     d) 在线难度估计消融:
        - 完整系统 (在线EMA)
        - 离线难度估计 (原始TimeLens方案)
        - 对比指标: 训练时间、性能稳定性

  3. 单一变量控制:
     - 每次消融只改变一个组件
     - 保持随机种子一致
     - 相同的数据预处理流程

AcceptanceCriteria:
  1. 基线对比: 相比原始TimeLens，R1@0.5提升 > 3%
  2. 连续嵌入消融: 高频动作(>30fps)精度提升 > 8%
  3. 时序适配器消融: 长视频(>5min)mIoU提升 > 5%
  4. 多维奖励消融: 组合奖励优于单一奖励 > 5%
  5. 在线难度消融: 训练时间减少 > 25%，性能不降
  6. 所有实验有完整的统计显著性检验 (p < 0.05)

Dependencies: [ARCH-003, ARCH-006, RL-004]
```

### Task 3.2: 性能与显存Profiling

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
  - Report: EVAL-002-performance-profiling.pdf
  - Data: profiling_results/latency_breakdown.json
  - Data: profiling_results/memory_timeline.json
  - Visualization: profiling_plots/ (火焰图、内存曲线)
  - Recommendations: EVAL-002-optimization-suggestions.md

ProfilingPlan:
  1. 延迟Profiling:

     a) 端到端推理延迟:
        - 输入: 30s, 60s, 300s 视频
        - 测量: 预处理 -> 视觉编码 -> 时序适配 -> LLM -> 输出
        - 指标: P50, P95, P99 延迟
        - 目标: P95 < 2x 视频时长 (2倍速)

     b) 模块级延迟分解:
        - Visual Encoder: Qwen3 ViT
        - Temporal Adapter: Mamba blocks
        - LLM: Qwen3 Decoder
        - 后处理: 时间戳提取
        - 找出瓶颈模块

  2. 显存Profiling:

     a) 峰值显存分析:
        - 测量完整训练流程的显存时间线
        - 识别显存峰值点 (通常是前向/后向传播)
        - 分解: 模型参数、激活值、优化器状态、临时缓存

     b) 组件级显存占用:
        - Qwen3 ViT: X GB
        - Temporal Adapter: Y GB (目标 < 2GB)
        - Qwen3 LLM: Z GB
        - 激活值缓存: W GB

     c) 验证L40适配:
        - 在L40 48GB上测量
        - 确保峰值 < 40GB (留8GB余量)
        - 如有超出，定位原因并优化

  3. 训练速度Profiling:
     - 每秒处理样本数 (samples/sec)
     - 对比: 在线难度估计 vs 离线难度估计
     - 验证: 在线方案训练时间减少 > 25%

  4. 并发能力测试:
     - 在4x L40上测试多路并发
     - 测量: 2路、4路并发时的延迟和吞吐量
     - 目标: 支持至少2路并发，每路延迟增加 < 50%

AcceptanceCriteria:
  1. 端到端P95延迟 < 2x视频时长 (所有测试视频长度)
  2. 峰值显存 < 40GB (L40 48GB，留8GB余量)
  3. Temporal Adapter显存占用 < 2GB
  4. 在线vs离线训练速度提升 > 25%
  5. 2路并发延迟增加 < 50%
  6. 完整的Profiling报告，包含瓶颈分析和优化建议

Dependencies: [ARCH-006, RL-003]
```

---

## Agent协作流程

### 协作模式

```
周1-2: [ArchitectureAgent] Phase 1 架构设计
         ↓ (交付: 接口定义)
周2-3: [ArchitectureAgent] 连续时间嵌入实现
         ↓ (交付: 可运行的嵌入模块)
周3-4: [ArchitectureAgent] 时序适配器实现
         ↓ (交付: 完整的Adapter + 导出功能)
周4-5: [RLAgent] Phase 2 奖励函数设计
         ↓ (交付: 奖励接口定义)
周5-6: [RLAgent] 多维奖励实现 + 在线难度估计
         ↓ (交付: 可运行的RL管道)
周6-7: [RLAgent] 分层采样实现
         ↓ (交付: 完整的RLVR升级)
周8-10: [EvaluationAgent] Phase 3 评估
         ↓ (交付: 完整评估报告)
周10-12: 所有Agent联合调试与文档整理
```

### 关键协作点

1. **ArchitectureAgent → RLAgent**:
   - 交付: `ContinuousTemporalEmbedding` 模块接口
   - 交付: `TemporalAdapter` 模块及导出功能
   - 时间: Phase 1第4周末

2. **RLAgent → EvaluationAgent**:
   - 交付: 完整的RLVR训练管道
   - 交付: 配置文件模板
   - 时间: Phase 2第7周末

3. **EvaluationAgent → All**:
   - 交付: 性能瓶颈分析报告
   - 交付: 优化建议
   - 时间: Phase 3第10周

---

## 项目里程碑

| 周 | 里程碑 | 交付物 | 负责Agent |
|---|-------|--------|----------|
| 2 | M1: 架构设计完成 | 接口定义文档、模块设计图 | ArchitectureAgent |
| 4 | M2: 连续时间嵌入完成 | 可运行的CTEModule | ArchitectureAgent |
| 5 | M3: 时序适配器完成 | TemporalAdapter + ONNX导出 | ArchitectureAgent |
| 6 | M4: 奖励函数框架完成 | 奖励接口、CLIP集成 | RLAgent |
| 7 | M5: 在线难度估计完成 | 完整的RL管道 | RLAgent |
| 8 | M6: 分层采样完成 | 完整的RLVR升级 | RLAgent |
| 10 | M7: 评估完成 | 完整的评估报告 | EvaluationAgent |
| 12 | M8: 项目完成 | 完整文档、Demo | All |

---

## 风险与缓解策略

| 风险 | 可能性 | 影响 | 缓解策略 |
|-----|-------|------|---------|
| Mamba适配器性能不达预期 | 中 | 高 | 准备LSTM备份方案；提前做小规模验证 |
| CLIP奖励计算太慢 | 中 | 中 | 使用更小的SigLIP；缓存特征 |
| L40显存不足 | 高 | 高 | 准备梯度检查点；降低batch size |
| 多Agent协作不畅 | 低 | 高 | 每周同步会议；清晰的接口文档 |
| 评估周期太长 | 中 | 中 | 准备快速评估子集；并行跑实验 |

---

## 附录

### A. 代码规范

```python
# Agent代码必须遵循的规范

class AgentModule:
    """所有Agent模块的基类规范"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *args, **kwargs) -> Tensor:
        """前向传播必须有类型注解"""
        raise NotImplementedError

    def get_memory_footprint(self) -> Dict[str, float]:
        """返回显存占用统计 (MB)"""
        return {
            'parameters': self._count_params(),
            'activations': self._estimate_activation_memory(),
            'total': self._count_total_memory()
        }

    def export_onnx(self, path: str, **kwargs) -> bool:
        """必须支持ONNX导出"""
        try:
            torch.onnx.export(self, ...)
            return True
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            return False
```

### B. 接口契约

```python
# ArchitectureAgent -> RLAgent 接口

class TemporalEmbeddingInterface(Protocol):
    """时间嵌入模块接口"""

    def encode_timestamp(self, timestamp: float) -> Tensor:
        """
        Args:
            timestamp: 秒级浮点时间戳
        Returns:
            Tensor: [hidden_dim] 连续时间嵌入
        """
        ...

    def decode_temporal_embedding(self, embedding: Tensor) -> float:
        """可选: 从嵌入解码回时间戳 (用于可解释性)"""
        ...

# RLAgent -> EvaluationAgent 接口

class RLVRPipelineInterface(Protocol):
    """RLVR训练管道接口"""

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        执行一步GRPO训练
        Returns:
            metrics: 包含reward, loss, kl_div等
        """
        ...

    def get_sample_weights(self, sample_ids: List[str]) -> Tensor:
        """获取样本的当前难度权重"""
        ...

    def export_checkpoint(self, path: str) -> None:
        """导出训练状态检查点"""
        ...
```

### C. 数据流转

```
原始视频数据
    │
    ▼
[DataProcessingAgent]
    │ 预处理: 抽帧、缩放、时间戳对齐
    ▼
Dataset (VideoTensor, Query, GroundTruthTime)
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
[ArchitectureAgent]          [RLAgent - 可选]
    │ 时间编码                    │ 难度分析
    │                              │
    ▼                              ▼
TemporalEmbeddedFeatures    Difficulty-Weighted Sampling
    │                              │
    └──────────────────────────────┘
                   │
                   ▼
[ArchitectureAgent]
    │ 时序适配器 + LLM推理
    ▼
PredictedTimeSpann
    │
    ▼
[RLAgent]
    │ 奖励计算 (IoU + 语义)
    ▼
RewardSignal → GRPO Update
    │
    ▼
[EvaluationAgent]
    │ 指标计算 (R1@0.5, mIoU)
    ▼
PerformanceMetrics
```

---

**文档版本**: v1.0
**最后更新**: 2025-01-XX
**维护者**: ProjectManager Agent
