# 独立口语改写数据集与离线中文编码器设计

日期：2026-08-14

## 1. 背景与目标

现有细分动作入口包含 254 个意图标签：49 个多设备协同场景和 205 个对象细分操作。字符 n-gram 模型在开发测试上可工作，但最终独立 Blind v4 的 scene/object/overall 准确率分别为 80.39%、53.14%、57.93%，不足以支持真实口语命令。

本次目标是建立约 5,000 条独立标注的中文口语数据，下载并离线部署中文预训练文本编码器，通过训练、验证、独立测试和最终新盲测，使完整解析器达到：

- 可判定指令 scene top-1 >= 95%；
- 可判定指令 object top-1 >= 95%；
- 可判定指令 overall top-1 >= 95%；
- 否定、问句、完成体陈述、歧义、不兼容操作和域外输入安全拒识率 100%；
- 所有拒识结果不含动作字段；
- 运行时不依赖网络。

## 2. 模型选择

默认编码器为 `BAAI/bge-small-zh-v1.5`：

- 使用官方 Hugging Face 仓库的固定 revision；
- 只下载 `model.safetensors`、配置和 tokenizer，不下载 pickle 权重；
- 权重约 95.8 MB，隐藏维度 512，4 层 Transformer；
- MIT 许可；
- 下载后生成 manifest，记录仓库、revision、文件大小、逐文件 SHA-256、许可证和下载时间；
- 默认从项目本地目录 `models/pretrained/BAAI_bge-small-zh-v1.5/` 以 `local_files_only=True` 加载。

若固定 revision 下载失败或官方文件不完整，不自动换用未审核模型；任务停止并报告错误。`moka-ai/m3e-small` 因模型卡说明仅限非商用而不采用。`bge-base-zh-v1.5` 作为后备实验选项，不在首轮下载范围内。

## 3. 独立标注数据设计

### 3.1 正例

统一目录的每个标签生成 16 条语义正确且句式实质不同的口语表达：

- train：每标签 10 条，共 2,540 条；
- dev：每标签 3 条，共 762 条；
- independent test：每标签 3 条，共 762 条；
- 正例合计 4,064 条。

三部分由互不复用上下文的独立标注子代理完成。每条记录至少包含：

```json
{
  "sample_id": "train-object-original-air_conditioner-001",
  "text": "屋里有点闷，把空调调到舒服些",
  "label_id": "object:original:air_conditioner",
  "result_type": "object_detailed_action",
  "annotation_source": "independent_annotator_train",
  "object_mentions": ["空调"],
  "operation_evidence": ["调到舒服些"],
  "safety_class": null
}
```

标注要求：

- 不能只更换“请、麻烦、帮我”等礼貌前后缀；
- 必须改变语序、动词、目的、状态描述、对象指代或场景表达；
- 对象指令必须包含足以判定用途/操作的证据；
- 同名对象的不同任务必须用操作语义区分；
- 多设备场景必须表达整体目的，不能只罗列目录对象；
- 不得复制 catalog canonical command、旧 train/dev、Blind v2/v3/v4 的完整文本；
- 不得读取 Blind v2/v3/v4 原始指令来编写新样本。

### 3.2 安全与边界样本

另建立 900 条安全与边界样本，每类 100 条；每类固定分为 train 60 条、dev 20 条、independent test 20 条。与 4,064 条标签正例合计形成 4,964 条数据：

1. 取消性否定；
2. 限制性否定与词内边界正例；
3. 信息问句；
4. 完成体陈述；
5. 同名对象用途歧义；
6. 对象与操作不兼容；
7. 域外指令；
8. 低置信/近候选；
9. 多分句、转折与纠正。

应拒识的记录使用 `label_id: null` 和 `result_type: clarification_required`。限制性否定、词内边界以及转折后仍有明确主动作的可执行记录使用真实标签和成功 `result_type`。所有记录都带稳定 `safety_class` 和 `expected_executable`，禁止用“一律拒识”获得虚高安全指标。

### 3.3 独立性与质量门禁

构建器必须拒绝以下数据：

- sample_id 或规范化全文重复；
- 同一规范化文本跨标签；
- train/dev/test 之间完整文本重复；
- 同标签跨集合共享超限长连续 n-gram 或同一完整表达骨架；
- 标签不在 254 标签目录中；
- 正例缺少操作证据；
- 负例包含动作标签或预期动作；
- 任一标签正例数量不足；
- annotation_source 与所属 split 不一致。

自动检查之外，每个标注子代理输出逐标签审核报告；独立审查代理抽查全部 254 标签，发现错标时退回标注代理修订，不能由训练实现者直接改写测试集。

## 4. 模型结构与训练

### 4.1 输入与编码

- 使用本地 BGE tokenizer，最大长度 96；
- 使用 attention-mask mean pooling 并做 L2 normalization；
- 不使用检索 instruction 前缀，因为输入与标签原型均为短中文命令；
- 标签原型由 train 正例编码聚合，dev/test 原文不进入原型。

### 4.2 双头模型

模型包含：

- 254 类意图分类头；
- 2 类执行/拒识门控头；
- 编码向量同时用于监督对比损失，使同标签不同表达靠近、相近标签拉开。

训练分两阶段：

1. 冻结 BGE，训练分类头与门控头，确认数据和接口正确；
2. 解冻最后两层，以较低学习率联合训练。

总损失为意图交叉熵、门控交叉熵和监督对比损失的加权和。随机种子、batch size、学习率、epoch、最优 checkpoint 选择规则必须写入训练报告。模型选择只看 dev，不读取 independent test。

### 4.3 校准与拒识

在 dev 上校准：

- 执行门控阈值；
- top-1 最低置信度；
- top-1/top-2 最小 margin；
- scene/object 分组阈值（只有 dev 证明需要时才允许不同）。

阈值选择目标是在安全拒识 100% 的前提下最大化可判定准确率。Independent test 只运行一次并冻结结果。

## 5. 运行时集成

新增独立 BGE 模型加载与推理模块，`DetailedActionParser` 的公开接口不变。

数据流：

1. 通用语法层识别明显取消、信息问句和非执行陈述；
2. 目录对象/任务同义词产生候选集合；
3. BGE 双头模型对全部标签评分，并在候选内重排；
4. 门控、绝对置信度和 margin 共同决定成功或澄清；
5. 成功后只从统一目录读取额外细分动作。

本地 BGE checkpoint 是默认模型。原字符模型保留为兼容回退，但只有用户显式传 `--semantic-backend character` 时启用；缺失 BGE 文件时默认入口应清楚报错，不静默降级到低准确率模型。

## 6. 下载与离线安全

- 下载发生在显式模型安装脚本中，不在 GUI 启动时自动联网；
- 使用固定 Hugging Face revision；
- 优先下载 safetensors；
- 下载目录使用临时 staging，文件齐全并校验后再原子移动到正式目录；
- manifest 和许可证随模型保存；
- 测试通过 `local_files_only=True` 和断网模拟证明运行期不联网；
- 不删除现有模型、数据或用户文件。

## 7. 测试与盲测流程

### 7.1 数据测试

- 254 标签覆盖和正例数量；
- 900 条安全/边界样本按 9 类和 60/20/20 split 精确覆盖；
- schema、唯一性、文本/骨架/长 n-gram 隔离；
- catalog、数据文件和 manifest SHA 一致；
- 测试作者与训练作者隔离。

### 7.2 模型测试

- 本地模型加载，不触网；
- pooling、batch 边界、标签顺序、checkpoint SHA；
- frozen-head 红绿测试；
- 同 seed 可复现；
- dev 阈值校准只使用 dev。

### 7.3 Independent test

在模型和阈值冻结后只运行一次：

- scene/object/overall top-1 均 >= 95%；
- 九类安全/边界各 100%；
- 无动作泄漏；
- 失败 ID、实际标签、置信度和 margin 全量留档。

### 7.4 最终 Blind v5

由未接触训练数据、模型实现和 independent test 的新代理，在代码冻结后编写并冻结 Blind v5，再进行一次黑盒评测。Blind v5 至少覆盖 49 scene、205 object、120 安全/边界和 24 模型参与 probes。通过门槛与 independent test 相同；文件、cases、catalog、model 和 evidence 均记录 SHA-256。

若 Blind v5 不达标，停止调参并如实报告；不得读取 Blind v5 原句后修改运行时再继续声称其为盲测。

## 8. 产物

- `datasets/detailed_action_spoken_v2/train.json`
- `datasets/detailed_action_spoken_v2/dev.json`
- `datasets/detailed_action_spoken_v2/independent_test.json`
- `datasets/detailed_action_spoken_v2/manifest.json`
- `models/pretrained/BAAI_bge-small-zh-v1.5/`
- `models/detailed_action_bge/best.pt`
- `models/detailed_action_bge/training_report.json`
- `tools/detailed_action_bge_model.py`
- `tools/detailed_action_spoken_dataset.py`
- `train_detailed_action_bge.py`
- 更新后的 `tools/detailed_action_runtime.py`
- 数据、模型、运行时、CLI、隔离和盲测测试文件与报告。

## 9. 错误处理

- 模型下载失败：保留 staging 和错误日志，不替换正式模型；
- SHA/文件缺失：拒绝加载并指出文件；
- CUDA OOM：自动降低 batch size 并从本 epoch 重新开始，记录调整；不自动改模型或数据；
- 标注校验失败：不进入训练；
- independent test 或 Blind v5 未达标：保留 checkpoint、证据和失败清单，项目状态为未达标；
- 旧入口继续保持无细分动作字段。

## 10. 非目标

- 不修改 254 标签目录含义；
- 不把 Blind v2/v3/v4 文本加入训练；
- 不增加在线推理服务；
- 不恢复旧入口中的细分动作；
- 不通过完整命令到标签的硬编码映射提高指标；
- 不删除现有字符模型或历史证据。
