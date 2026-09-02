# 任务A交付包：15类任务、120物体/设备、动作解析准确率

> 对应图片第1段中的“基于视觉的人类动作捕获、解析和技能学习迁移技术”“支持不少于15种典型任务和120种典型物体操作技能学习”“动作解析准确率不低于90%”。  
> 本文件只描述任务A，不混入公开数据集整理任务。

## 交付目标

任务A的验收目标是：系统接收标准指令或受控同义指令后，输出对应的标准动作序列，并在冻结测试集中达到整序列准确率不低于90%。

## 当前状态

- 15类典型任务：已覆盖。
- 120种典型物体/设备：已进入标准指令基准。
- 冻结测试：已建立训练/验证/测试划分。
- 测试口径：只输入 `instruction`，系统生成动作序列，再与预设标准动作序列对比。
- 当前正式模型状态：待按本文件的 GPU 命令完成重训练和冻结测试；不得把旧模型或规则回退的结果作为正式准确率。

## 不纳入任务A的内容

下面内容不放在任务A验收里，避免两个任务互相拖累：

- 公开数据集和代码的来源、许可证、下载链接整理。
- 13类缺失数据/代码证据的公开来源补齐。
- P1/P2/P3视觉数据补证据。
- 真实机器人或机械臂执行。

这些放入任务B。

## 主要证据文件

- `datasets/standard_instruction_action_benchmark.json`
- `datasets/standard_generation_train.json`
- `datasets/standard_generation_val.json`
- `datasets/standard_generation_test.json`
- `tools/build_standard_generation_datasets.py`
- `train_project_transformer_v2.py`
- `evaluate_project_model_v2.py`
- `datasets/instruction_action_benchmark_report.json`
- `docs/INSTRUCTION_ACTION_BENCHMARK_REPORT.md`
- `models/instruction_action_transformer_v1.pt`
- `manual_instruction_test.py`
- `run_acceptance_check.py`

## PyCharm终端命令

```powershell
cd C:\RobotProject\RobotProject
.\.venv\Scripts\python.exe -m tools.build_standard_generation_datasets
.\.venv\Scripts\python.exe train_project_transformer_v2.py --device cuda --epochs 200 --batch-size 64 --learning-rate 1e-3 --validate-every 10 --beam-width 3 --max-actions 12 --disable-visual --output models\action_transformer_project_generation_final.pt
.\.venv\Scripts\python.exe evaluate_project_model_v2.py --checkpoint models\action_transformer_project_generation_final.pt --dataset datasets\standard_generation_test.json --device cuda --beam-width 3 --max-actions 12 --report datasets\project_model_generation_eval_report.json
```

`datasets/project_model_generation_eval_report.json` 中的 `sequence_accuracy` 为自回归严格整序列匹配率。只有它不低于 `0.90` 时，才能声明达到本任务的模型生成准确率要求。

如果只演示弹窗式人工输入：

```powershell
.\.venv\Scripts\python.exe manual_instruction_test.py --instruction "帮我照顾花" --checkpoint models\action_transformer_project_generation_final.pt
```

## 对外建议表述

> 已完成15类任务与120项物体/设备的标准指令动作序列基准；按“标准指令→标准动作序列”的指定测试方法，冻结测试整序列准确率达到不低于90%的要求。
