# 细分动作随机口语可视化测试入口实施计划

**目标：** 建立运行时动态生成口语命令的独立细分动作可视化测试入口，逐条计时并验证总体准确率 >80%。

**架构：** 将动态用例生成、执行判定、证据输出与 Tkinter 展示拆分。纯逻辑模块可在无 GUI 环境中测试；入口同时支持 GUI 和 `--no-gui`，便于自动验收。

**技术栈：** Python、Tkinter、`DetailedActionParser`、JSON/JSONL/CSV、`unittest`/pytest。

---

### 任务 1：运行时口语用例生成器

**文件：**
- 新建 `tools/detailed_action_random_cases.py`
- 新建 `tests/test_detailed_action_random_cases.py`

先写失败测试，覆盖数量、唯一性、随机种子复现、对象/场景覆盖、标准标签有效以及禁止复制正式场景命令。随后实现结构化语法槽组合器，只读取正式目录结构，不读取任何数据 split 文本。

### 任务 2：批量执行、判定与分类计时

**文件：**
- 新建 `tools/detailed_action_random_runner.py`
- 新建 `tests/test_detailed_action_random_runner.py`

先用伪解析器和可控计时器写失败测试，覆盖每条耗时、标签判定、异常处理、总体与分类平均值。随后实现 runner，并确保计时边界只包围 `resolve()`。

### 任务 3：可审计输出

**文件：**
- 新建 `tools/detailed_action_random_output.py`
- 新建 `tests/test_detailed_action_random_output.py`

先测试生成文件集合和关键字段，再实现时间戳运行目录、JSON、JSONL、CSV、Markdown 报告的增量写入与最终封存。

### 任务 4：CLI 与 Tkinter 可视化入口

**文件：**
- 新建 `detailed_action_random_visual_test.py`
- 新建 `tests/test_detailed_action_random_visual_test.py`

先测试参数校验、批次事件和 CLI 返回值，再实现参考 500-case 演示的响应式界面、结果筛选、实时单条耗时与三个平均耗时指标。

### 任务 5：回归与真实验收

运行新增单元测试、既有细分入口隔离测试和语法编译。随后以固定种子运行足量随机批次，检查命令无重复、证据文件完整且总体准确率严格大于 80%。若未达标，只依据生成器覆盖和错误类型修正规则或运行时，不读取冻结独立测试文本。

