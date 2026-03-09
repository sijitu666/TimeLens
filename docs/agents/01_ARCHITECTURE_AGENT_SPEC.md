# Architecture Agent 详细需求规范

## Agent信息

- **Agent名称**: ArchitectureAgent
- **代号**: ARCH
- **职责**: 架构设计与时间编码重构 (Phase 1)
- **协作Agent**:
  - 上游: 无 (基础Agent)
  - 下游: RLAgent, EvaluationAgent
- **工期**: 5周 (Week 1-5)

---

## 工作空间

### 负责目录

```
timelens-advanced/
├── training/
│   ├── models/                    # [ARCH负责] 新增模型组件
│   │   ├── __init__.py
│   │   ├── continuous_temporal_embedding.py    # Task 1.1.2
│   │   ├── temporal_rope.py                   # Task 1.1.2
│   │   ├── temporal_adapter.py               # Task 1.2.1
│   │   ├── mamba_temporal_adapter.py         # Task 1.2.2
│   │   └── mamba_block.py                    # Task 1.2.2
│   │
│   └── train/                     # [ARCH修改] 训练脚本适配
│       ├── train_sft_timelens.py           # 集成CT Embedding
│       └── train_grpo_timelens.py          # 集成Adapter接口
│
├── inference/                     # [ARCH负责] 推理优化
│   ├── __init__.py
│   ├── streaming_vtg_pipeline.py            # Task 1.2.3
│   ├── onnx_exporter.py                    # Task 1.2.3
│   └── tensorrt_optimizer.py               # Task 1.2.3
│
├── tests/                         # [ARCH负责] 单元测试
│   ├── test_continuous_embedding.py
│   ├── test_temporal_adapter.py
│   ├── test_mamba_adapter.py
│   └── test_onnx_export.py
│
└── docs/architecture/             # [ARCH负责] 架构文档
    ├── designs/
    │   ├── ARCH-001-removal-spec.md
    │   ├── ARCH-002-embedding-design.md
    │   ├── ARCH-003-high-freq-optimization.md
    │   ├── ARCH-004-temporal-state-design.md
    │   ├── ARCH-005-mamba-adapter-impl.md
    │   └── ARCH-006-deployment-guide.md
    └── api/
        └── temporal-embedding-interface.md
```

### 接口文件

**必须实现的接口** (位于 `docs/agents/interfaces/`):

1. `interface_architecture.py` - ArchitectureAgent对外接口
2. `schema_architecture.json` - 数据格式Schema
3. `README_ARCHITECTURE.md` - 接口使用说明

---

## 任务清单

### Task 1.1: 连续时间嵌入

#### 1.1.1 移除文本时间戳预处理
```yaml
TaskID: ARCH-001
EstimatedTime: 3天
Priority: P0

Deliverables:
  - Modified: training/data/grounding.py
  - Modified: training/data/preprocess.py
  - Doc: ARCH-001-removal-spec.md

SuccessCriteria:
  1. 原始timestamp转text token逻辑完全移除
  2. 浮点时间戳格式保留并传递
  3. 单元测试通过率100%
```

#### 1.1.2 连续时间嵌入层
```yaml
TaskID: ARCH-002
EstimatedTime: 7天
Priority: P0

Deliverables:
  - New: training/models/continuous_temporal_embedding.py
  - New: training/models/temporal_rope.py
  - Modified: training/train/train_sft_timelens.py (集成点)
  - Test: tests/test_continuous_embedding.py
  - Doc: ARCH-002-embedding-design.md

TechnicalSpecs:
  hidden_dim: 3584  # Qwen3 hidden size
  time_range: [0.001, 10000]  # 秒
  precision: 0.001  # 毫秒
  methods: [MLP, 1D-RoPE]

SuccessCriteria:
  1. 微秒级精度支持
  2. 梯度正常传播
  3. 合成数据验证时序感知
```

#### 1.1.3 高频动作优化
```yaml
TaskID: ARCH-003
EstimatedTime: 4天
Priority: P1

Deliverables:
  - Modified: training/models/continuous_temporal_embedding.py
  - New: training/models/high_freq_temporal_attention.py
  - Test: tests/test_high_freq_capture.py
  - Doc: ARCH-003-high-freq-optimization.md

TechnicalSpecs:
  coarse_scale: 1.0  # 秒级
  fine_scale: 0.001  # 毫秒级
  fusion: [concat, add, gate]

SuccessCriteria:
  1. <0.1s时间差区分
  2. 高频数据集IoU提升>5%
  3. 计算overhead <15%
```

### Task 1.2: 轻量级时序适配器

#### 1.2.1 时序状态传递机制
```yaml
TaskID: ARCH-004
EstimatedTime: 5天
Priority: P0

Deliverables:
  - New: training/models/temporal_adapter.py
  - New: training/models/temporal_state_bank.py
  - Doc: ARCH-004-temporal-state-design.md
  - Diagram: temporal-adapter-arch.png

TechnicalSpecs:
  adapter_position: post_visual_merger_pre_llm
  state_dim: 256
  mechanism: [cross_attention, gated_residual]

SuccessCriteria:
  1. 消除显式时间Token
  2. 上下文长度减少>30%
  3. 短序列精度不降
```

#### 1.2.2 Mamba时序适配器实现
```yaml
TaskID: ARCH-005
EstimatedTime: 6天
Priority: P1

Deliverables:
  - New: training/models/mamba_temporal_adapter.py
  - New: training/models/mamba_block.py
  - Test: tests/test_mamba_adapter.py
  - Doc: ARCH-005-mamba-adapter-impl.md
  - ExportTest: test_onnx_export.py

TechnicalSpecs:
  d_model: 512
  d_state: 64
  d_conv: 3
  expand_factor: 2
  max_seq_len: 2048

SuccessCriteria:
  1. 参数量<10M
  2. 推理速度>1000 tokens/sec (A100)
  3. 成功导出ONNX
  4. 长序列建模提升>20%
```

#### 1.2.3 流式推理适配与ONNX导出
```yaml
TaskID: ARCH-006
EstimatedTime: 5天
Priority: P1

Deliverables:
  - New: inference/streaming_vtg_pipeline.py
  - New: inference/onnx_exporter.py
  - New: inference/tensorrt_optimizer.py
  - Doc: ARCH-006-deployment-guide.md
  - Benchmark: latency_benchmark.json

TechnicalSpecs:
  chunk_size: 30s
  overlap: 5s
  max_latency: 2x_video_duration
  first_packet_latency: 5s

SuccessCriteria:
  1. P95延迟<2x视频时长
  2. 首包延迟<5s
  3. ONNX与PyTorch差距<20%
  4. 显存占用<40GB
  5. 支持2路并发
```

---

## 接口契约

### 对外接口

**ArchitectureAgent → RLAgent**:
```python
class TemporalEmbeddingInterface:
    def encode_timestamp(self, timestamp: float) -> Tensor:
        """将浮点时间戳编码为连续嵌入"""
        ...

class TemporalAdapterInterface:
    def forward(self, visual_features: Tensor, temporal_states: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """时序适配器前向，返回增强特征和更新状态"""
        ...
```

**ArchitectureAgent → EvaluationAgent**:
```python
class ModelExportInterface:
    def export_onnx(self, path: str, opset_version: int = 14) -> bool:
        """导出ONNX格式"""
        ...

    def get_memory_footprint(self) -> Dict[str, float]:
        """获取显存占用统计"""
        ...
```

---

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|-----|-------|------|---------|
| Mamba适配器性能不达预期 | 中 | 高 | 准备LSTM备份方案；提前验证 |
| ONNX导出失败 | 中 | 高 | 提前测试关键算子；准备PyTorch部署备份 |
| 高频动作精度提升不足 | 中 | 中 | 增加更多合成数据；调整双尺度权重 |
| 与RLAgent接口不匹配 | 低 | 高 | 提前冻结接口；增加接口测试 |

---

## 附录

### A. 技术栈

- **深度学习框架**: PyTorch 2.6+
- **模型基础**: Qwen3-VL-8B
- **状态空间模型**: Mamba-2 (选择性扫描)
- **推理优化**: ONNX Runtime, TensorRT
- **流式处理**: Redis Streams / Kafka (可选)

### B. 参考资源

- Mamba论文: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- TimeLens论文: "TimeLens: Rethinking Video Temporal Grounding"
- Qwen3-VL技术报告
- ONNX导出最佳实践指南

---

**文档结束**

*本文档由ArchitectureAgent维护，所有Agent必须遵守*
