# Data Processing Agent 详细需求规范

## Agent信息

- **Agent名称**: DataProcessingAgent
- **代号**: DATA
- **职责**: 数据预处理管道改造、数据集管理、采样策略实现
- **协作Agent**:
  - 上游: 无 (基础数据服务)
  - 下游: ArchitectureAgent, RLAgent, EvaluationAgent (提供数据)
- **工期**: 贯穿整个项目 (Week 1-12)

---

## 工作空间

### 负责目录

```
timelens-advanced/
├── data/                          # [DATA负责] 数据集目录
│   ├── raw/                       # 原始数据
│   │   ├── videos/                # 原始视频文件
│   │   └── annotations/           # 原始标注
│   ├── processed/                 # 处理后数据
│   │   ├── frames/                # 抽帧结果
│   │   ├── features/              # 预提取特征
│   │   └── metadata/              # 元数据
│   └── cache/                     # 缓存数据
│       ├── dataloader_cache/
│       └── sampler_cache/
│
├── timelens/dataset/              # [DATA修改] 数据集模块
│   ├── __init__.py
│   ├── timelens_data.py           # 基础数据集类
│   ├── advanced_dataset.py        # DATA-001: 高级数据集
│   ├── streaming_dataset.py       # DATA-004: 流式数据集
│   └── samplers/                  # 采样器
│       ├── __init__.py
│       ├── difficulty_aware_sampler.py    # DATA-003
│       ├── stratified_sampler.py          # DATA-003
│       └── gaussian_sampler.py            # 原始TimeLens
│
├── training/data/                 # [DATA修改] 训练数据模块
│   ├── __init__.py
│   ├── grounding.py               # 修改支持连续时间
│   ├── hybrid.py                  # 混合数据集
│   ├── preprocess.py              # 预处理管道
│   ├── collator.py                # 数据收集器
│   └── advanced_collator.py     # DATA-002: 高级收集器
│
├── preprocessing/                 # [DATA负责] 预处理工具
│   ├── __init__.py
│   ├── video_processor.py         # 视频处理
│   ├── frame_extractor.py         # 抽帧工具
│   ├── annotation_parser.py       # 标注解析
│   └── metadata_builder.py        # 元数据构建
│
├── tests/data/                    # [DATA负责] 数据测试
│   ├── test_datasets.py
│   ├── test_samplers.py
│   ├── test_preprocess.py
│   └── test_streaming.py
│
└── docs/agents/data/              # [DATA负责] 数据文档
    ├── README.md
    ├── DATASET_GUIDE.md
    ├── PREPROCESSING.md
    └── api/
        └── dataset-interface.md
```

### 接口文件

**必须实现的接口** (位于 `docs/agents/interfaces/`):

1. `interface_data.py` - DataProcessingAgent对外接口
2. `schema_data.json` - 数据格式Schema
3. `README_DATA.md` - 接口使用说明

---

## 任务清单

### Task: 数据预处理管道改造

#### DATA-001: 高级数据集类实现

```yaml
TaskID: DATA-001
Title: 实现支持高级特性的数据集类
Agent: DataProcessingAgent
Priority: P0
EstimatedTime: 5天
Input:
  - 原始TimeLens数据集类
  - ArchitectureAgent的连续时间嵌入需求
  - RLAgent的难度感知采样需求
Output:
  - NewFile: timelens/dataset/advanced_dataset.py
  - ModifiedFile: timelens/dataset/timelens_data.py (向后兼容)
  - TestFile: tests/data/test_advanced_dataset.py
  - Doc: docs/agents/data/DATA-001-spec.md
TechnicalSpecs:
  features:
    continuous_time_support:
      description: "支持浮点时间戳，不强制转换为文本"
      data_type: "float32"
      precision: "millisecond"
    difficulty_aware:
      description: "支持难度分数关联"
      storage: "in-memory hash map"
      update_mode: "online"
    streaming_ready:
      description: "支持流式数据加载"
      chunk_size: "configurable"
      prefetch: "enabled"
    multi_resolution:
      description: "支持多分辨率视频"
      resolutions: ["360p", "480p", "720p", "1080p"]
      selection: "adaptive based on video length"
  data_schema:
    sample_fields:
      - name: "video_path"
        type: "str"
        description: "视频文件路径"
      - name: "query"
        type: "str"
        description: "文本查询"
      - name: "ground_truth_span"
        type: "Tuple[float, float]"
        description: "真实时间区间 (秒)"
      - name: "timestamp_continuous"
        type: "float32"
        description: "连续时间戳 (支持毫秒)"
      - name: "difficulty_score"
        type: "float32"
        description: "难度分数 [0, 1]"
        optional: true
      - name: "metadata"
        type: "Dict[str, Any]"
        description: "额外元数据"
        optional: true
  backward_compatibility:
    requirement: "保持与原始TimeLens数据集API兼容"
    migration: "提供自动迁移工具"
SuccessCriteria:
  - continuous_time_precision: "millisecond level support"
  - difficulty_score_update: "< 10ms latency"
  - data_loading_throughput: "> 100 samples/sec"
  - backward_compatible: "existing code runs without modification"
  - unit_test_coverage: "> 90%"
Dependencies: []
```

#### DATA-002: 高级数据收集器

```yaml
TaskID: DATA-002
Title: 实现支持连续时间和动态批次的数据收集器
Agent: DataProcessingAgent
Priority: P0
EstimatedTime: 4天
Input:
  - 原始TimeLens数据收集器
  - ArchitectureAgent的连续时间嵌入需求
  - RLAgent的动态采样需求
Output:
  - NewFile: training/data/advanced_collator.py
  - ModifiedFile: training/data/collator.py (向后兼容)
  - TestFile: tests/data/test_advanced_collator.py
  - Doc: docs/agents/data/DATA-002-spec.md
TechnicalSpecs:
  features:
    continuous_time_batching:
      description: "批次内保持浮点时间戳"
      padding: "not applicable for continuous values"
      tensor_type: "float32"
    dynamic_batch_size:
      description: "根据序列长度动态调整批次大小"
      strategy: "bucket by sequence length"
      memory_limit: "configure max memory per batch"
    variable_resolution:
      description: "支持批次内不同分辨率视频"
      handling: "pad to max resolution in batch"
    metadata_preservation:
      description: "保留样本元数据通过批处理"
      storage: "side information tensor"
  batch_schema:
    fields:
      - name: "video_frames"
        type: "Tensor[float16]"
        shape: "[B, T, C, H, W]"
        description: "视频帧批次"
      - name: "video_masks"
        type: "Tensor[bool]"
        shape: "[B, T]"
        description: "视频帧掩码 (处理变长)"
      - name: "queries"
        type: "List[str]"
        length: "B"
        description: "文本查询列表"
      - name: "ground_truth_spans"
        type: "Tensor[float32]"
        shape: "[B, 2]"
        description: "真实时间区间 [start, end]"
      - name: "continuous_timestamps"
        type: "Tensor[float32]"
        shape: "[B, T]"
        description: "连续时间戳 (用于每帧)"
      - name: "difficulty_scores"
        type: "Tensor[float32]"
        shape: "[B]"
        description: "样本难度分数"
        optional: true
      - name: "sample_ids"
        type: "List[str]"
        length: "B"
        description: "样本ID列表 (用于追踪)"
  dynamic_batching_strategy:
    bucketing:
      num_buckets: 8
      bucket_boundaries: "based on sequence length distribution"
    batch_size_calculation: |
      effective_batch_size = min(
          configured_batch_size,
          max_memory // memory_per_sample
      )
    padding_strategy: "pad to max length in batch (per bucket)"
SuccessCriteria:
  - batching_latency: "< 50ms per batch"
  - memory_efficiency: "< 10% padding overhead"
  - continuous_time_preservation: "no precision loss"
  - dynamic_batch_correctness: "verified by memory profiling"
  - backward_compatible: "existing collators still work"
Dependencies: [DATA-001]
```

#### DATA-003: 难度感知采样器

```yaml
TaskID: DATA-003
Title: 实现难度感知和分层采样器
Agent: DataProcessingAgent
Priority: P0
EstimatedTime: 5天
Input:
  - 原始TimeLens采样器
  - RLAgent的在线难度估计需求
  - RLAgent的分层采样需求
Output:
  - NewFile: timelens/dataset/samplers/difficulty_aware_sampler.py
  - NewFile: timelens/dataset/samplers/stratified_sampler.py
  - ModifiedFile: timelens/dataset/samplers/gaussian_sampler.py (增强)
  - TestFile: tests/data/test_samplers.py
  - Doc: docs/agents/data/DATA-003-spec.md
TechnicalSpecs:
  samplers:
    difficulty_aware_sampler:
      description: "根据难度分数调整采样概率"
      strategy: "higher difficulty -> higher sampling probability"
      probability_formula: |
        P(sample_i) ∝ (difficulty_i + ε)^γ
        where:
        - γ > 1: emphasize hard samples
        - γ = 1: linear
        - γ < 1: de-emphasize
      temperature: "configurable for annealing"
    stratified_sampler:
      description: "按难度分层后均匀采样"
      strata_definition:
        - name: "easy"
          range: "[0, 0.33)"
          proportion: "0.33"
        - name: "medium"
          range: "[0.33, 0.67)"
          proportion: "0.33"
        - name: "hard"
          range: "[0.67, 1.0]"
          proportion: "0.34"
      sampling_within_stratum: "uniform or Gaussian"
      dynamic_stratification: "update boundaries based on distribution"
    enhanced_gaussian_sampler:
      description: "原始TimeLens高斯采样的增强版"
      base: "original TimeLens Gaussian around ground truth"
      enhancement: "adjust μ and σ based on difficulty"
      formula: |
        μ_new = μ + α·difficulty
        σ_new = σ·(1 - β·difficulty)
        where α, β are configurable hyperparameters
  difficulty_source:
    online: "from RLAgent's online estimator (EMA)"
    offline: "pre-computed from previous training runs"
    hybrid: "combination of both"
  update_frequency:
    mode: "per epoch" or "per N steps"
    notification: "callback to sampler when difficulty updated"
SuccessCriteria:
  - sampler_latency: "< 10ms per batch"
  - difficulty_integration: "correctly read from RLAgent"
  - sampling_distribution: "matches intended probability"
  - stratification_correctness: "correct boundary enforcement"
  - gaussian_adjustment: "verified by statistical tests"
  - backward_compatible: "can disable difficulty awareness"
Dependencies: [RL-003, RL-004]
```

#### DATA-004: 流式数据集

```yaml
TaskID: DATA-004
Title: 实现支持流式处理的数据集
Agent: DataProcessingAgent
Priority: P1
EstimatedTime: 4天
Input:
  - ArchitectureAgent的流式处理需求
  - MLOps部署的长视频处理需求
Output:
  - NewFile: timelens/dataset/streaming_dataset.py
  - ModifiedFile: training/data/hybrid.py (流式支持)
  - TestFile: tests/data/test_streaming.py
  - Doc: docs/agents/data/DATA-004-spec.md
TechnicalSpecs:
  streaming_dataset:
    description: "支持流式加载长视频，避免内存爆炸"
    chunk_size: "configurable (default: 30s)"
    prefetch: "enabled"
    buffer_size: "configurable (default: 3 chunks)"
    overlap: "configurable for temporal continuity"
  features:
    lazy_loading:
      description: "按需加载视频块"
      trigger: "__getitem__ or iterator"
      caching: "LRU cache for recent chunks"
    memory_mapping:
      description: "使用内存映射读取大文件"
      benefit: "reduces memory footprint"
      support: "numpy.memmap or similar"
    dynamic_resolution:
      description: "根据可用内存动态调整分辨率"
      policy: "memory-based downscaling"
    streaming_transforms:
      description: "支持在流式加载时应用变换"
      operations: ["resize", "normalize", "augment"]
      location: "on-the-fly during loading"
  modes:
    training_mode:
      description: "训练模式，支持随机采样、Shuffle"
      characteristics: ["random access", "shuffle enabled", "multi-epoch"]
    inference_mode:
      description: "推理模式，支持顺序流式处理"
      characteristics: ["sequential access", "no shuffle", "single-pass"]
    hybrid_mode:
      description: "混合模式，短视频完整加载，长视频流式"
      threshold: "configurable duration threshold"
SuccessCriteria:
  - memory_efficiency: "supports >1 hour video on 48GB GPU"
  - loading_latency: "< 100ms per chunk"
  - throughput: "> 10 chunks/sec"
  - compatibility: "works with existing DataLoader"
  - mode_switching: "seamless train/inference switching"
  - error_handling: "graceful handling of corrupted chunks"
Dependencies: []
```

---

## 验收标准

### 功能验收

- [ ] 所有数据集类支持浮点时间戳 (精度: 毫秒)
- [ ] 数据收集器正确处理动态批次和连续时间
- [ ] 采样器正确集成难度分数，采样分布符合预期
- [ ] 流式数据集支持 >1小时视频，内存占用 <48GB
- [ ] 向后兼容: 原始TimeLens代码可无缝运行
- [ ] 单元测试覆盖率 > 90%

### 性能验收

- [ ] 数据加载吞吐量 > 100 samples/sec (batch_size=32)
- [ ] 批处理延迟 < 50ms per batch
- [ ] 流式加载延迟 < 100ms per chunk
- [ ] 内存效率: < 10% padding overhead
- [ ] 流式处理支持 >1小时视频

### 集成验收

- [ ] 与ArchitectureAgent接口兼容 (ContinuousTimeInterface)
- [ ] 与RLAgent接口兼容 (DifficultyAwareSamplingInterface)
- [ ] 与EvaluationAgent接口兼容 (DatasetEvaluationInterface)
- [ ] 所有接口测试通过率 100%
- [ ] 集成测试覆盖率 > 85%

---

## 文档与交付物

### 必须文档清单

1. **设计文档**:
   - `DATA-001-spec.md`: 高级数据集设计
   - `DATA-002-spec.md`: 高级收集器设计
   - `DATA-003-spec.md`: 难度感知采样器设计
   - `DATA-004-spec.md`: 流式数据集设计

2. **API文档**:
   - `docs/agents/interfaces/interface_data.py`: 数据接口定义
   - `docs/agents/interfaces/schema_data.json`: 数据格式Schema
   - `docs/agents/interfaces/README_DATA.md`: 接口使用说明

3. **开发指南**:
   - `docs/agents/data/README.md`: Agent概览
   - `docs/agents/data/DATASET_GUIDE.md`: 数据集使用指南
   - `docs/agents/data/PREPROCESSING.md`: 预处理指南

4. **测试报告**:
   - `tests/data/test_report.md`: 单元测试报告
   - `tests/data/coverage_report.html`: 覆盖率报告

---

**文档结束**

*本文档由DataProcessingAgent维护，版本变更请提交PR*
