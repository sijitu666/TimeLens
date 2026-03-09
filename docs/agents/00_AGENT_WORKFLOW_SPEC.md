# Agent 协作工作流规范

## 文档信息
- **版本**: v1.0
- **生效日期**: 2025-01-XX
- **维护者**: ProjectManager Agent

---

## 1. 工作模式

### 1.1 迭代开发周期

每个开发周期为 **1周**，遵循以下节奏：

```
周一: 周会 (同步进度、调整计划)
周二-周四: 独立开发 (Agent自主工作)
周五: 代码审查 (提交PR、互相Review)
周六: 集成测试 (合并到develop分支)
周日: 文档更新 (更新技术文档)
```

### 1.2 分支策略

```
main (稳定分支, 仅接受PR)
  ↑
develop (开发分支, 每周六集成)
  ↑
feature/arch-001 (ArchitectureAgent功能分支)
feature/rl-001   (RLAgent功能分支)
bugfix/eval-001  (EvaluationAgent修复分支)
```

**分支命名规范**:
- 功能分支: `feature/{agent-short-name}-{task-id}`
  - 例: `feature/arch-002-continuous-embedding`
- 修复分支: `bugfix/{agent-short-name}-{issue-id}`
  - 例: `bugfix/rl-001-reward-calc`

### 1.3 提交信息规范

**格式**: `<type>(<scope>): <subject>`

**类型 (type)**:
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具

**范围 (scope)**:
- `arch`: ArchitectureAgent相关
- `rl`: RLAgent相关
- `eval`: EvaluationAgent相关
- `data`: DataProcessingAgent相关
- `common`: 通用模块

**示例**:
```
feat(arch): implement continuous temporal embedding layer

- Add MLP-based time encoder
- Support microsecond precision
- Include gradient checkpointing

Refs: ARCH-002
```

---

## 2. 协作接口

### 2.1 Agent间通信协议

**接口定义文件位置**: `docs/agents/interfaces/`

每个Agent必须提供:
1. `interface_{agent_name}.py`: Python接口定义
2. `schema_{agent_name}.json`: JSON Schema数据格式
3. `README_{agent_name}.md`: 接口使用说明

### 2.2 数据交换格式

**标准数据包结构**:
```python
{
    "version": "1.0",
    "timestamp": "2025-01-15T10:30:00Z",
    "source_agent": "ArchitectureAgent",
    "target_agent": "RLAgent",
    "payload": {
        "type": "temporal_embeddings",
        "data": {...},  # 实际数据
        "metadata": {
            "shape": [32, 3584],
            "dtype": "float32"
        }
    }
}
```

### 2.3 依赖管理

**依赖声明文件**: `docs/agents/dependencies/{agent_name}_deps.yaml`

示例:
```yaml
agent: ArchitectureAgent
dependencies:
  upstream: []  # ArchitectureAgent是基础，无上游
  downstream:
    - agent: RLAgent
      interface: temporal_embedding_interface
      version: ">=1.0,<2.0"
    - agent: EvaluationAgent
      interface: model_evaluation_interface
      version: ">=1.0,<2.0"
```

---

## 3. 代码审查规范

### 3.1 审查清单

**功能性审查**:
- [ ] 实现符合接口定义
- [ ] 所有边界条件已处理
- [ ] 错误处理完善
- [ ] 单元测试覆盖 > 80%

**性能审查**:
- [ ] 显存使用符合预期
- [ ] 计算复杂度合理
- [ ] 批处理支持

**可读性审查**:
- [ ] 命名清晰一致
- [ ] 文档字符串完整
- [ ] 复杂逻辑有注释

### 3.2 审查流程

```
开发者提交PR
    ↓
自动化CI检查 (pytest, lint, type-check)
    ↓
Peer Review (至少1个其他Agent审查)
    ↓
Maintainer Review (ProjectManager或指定负责人)
    ↓
合并到develop分支
    ↓
周六集成测试
```

---

## 4. 文档规范

### 4.1 必须文档清单

**每个Agent必须维护**:
1. `README.md`: Agent简介、职责、快速开始
2. `ARCHITECTURE.md`: 架构设计文档
3. `API.md`: 接口API文档
4. `DEVELOPMENT.md`: 开发指南
5. `TROUBLESHOOTING.md`: 常见问题

**每个Task必须产出**:
1. 设计文档 (Task开始前)
2. 实现代码
3. 单元测试
4. 集成测试
5. 性能基准报告
6. 总结文档 (Task完成后)

### 4.2 文档模板

**设计文档模板** (`docs/templates/design_doc_template.md`):
```markdown
# [TaskID] 设计文档: [Title]

## 1. 概述
### 1.1 背景
### 1.2 目标
### 1.3 范围

## 2. 设计方案
### 2.1 架构图
### 2.2 关键组件
### 2.3 接口定义

## 3. 实现细节
### 3.1 算法描述
### 3.2 数据结构
### 3.3 错误处理

## 4. 测试计划
### 4.1 单元测试
### 4.2 集成测试
### 4.3 性能测试

## 5. 风险与缓解
## 6. 时间线
## 7. 附录
```

---

## 5. 应急响应

### 5.1 问题升级路径

**Level 1: Agent内部解决**
- 开发过程中的技术问题
- 由Agent自行解决或寻求Peer帮助

**Level 2: 跨Agent协调**
- 接口不匹配、依赖冲突
- 由ProjectManager协调

**Level 3: 项目级决策**
- 架构重大变更、工期调整
- 需要所有Agent负责人会议

### 5.2 快速响应清单

**当发现阻塞问题时**:
1. 立即在项目管理工具中标记阻塞状态
2. 通知相关Agent和ProjectManager
3. 提供问题描述、复现步骤、已尝试方案
4. 协商临时解决方案 (workaround)
5. 在问题解决后更新文档

---

**文档结束**

*本文档由ProjectManager Agent维护，所有Agent必须遵守。如有修改建议，请提交PR。*
