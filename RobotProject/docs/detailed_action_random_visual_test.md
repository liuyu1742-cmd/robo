# 细分动作随机口语可视化测试使用说明

在项目目录运行：

```powershell
.\.venv\Scripts\python.exe detailed_action_random_visual_test.py
```

也可以直接双击项目根目录的 `run_detailed_action_random_visual_test.bat`。该脚本固定使用项目 `.venv`，避免系统 Miniconda/Python 缺少 PyTorch 或 Transformers。

界面中可输入任意正整数测试条数和随机种子，并选择 `auto`、`cuda` 或 `cpu`。默认 `auto` 会优先使用可用 NVIDIA GPU；界面汇总区显示本次真正生效的设备。点击“生成随机预览”只生成命令，不加载模型；点击“开始测试”后后台加载本地 BGE，逐条展示识别结果、细分动作和解析耗时。

命令由运行时组合语法现场产生，不读取 train、dev 或冻结独立测试集文本。独立物体命令只组合目录允许的生活动作，多设备命令只使用已审核场景及其设备关系。

测试结束后界面显示总体准确率、独立物体平均耗时、多设备联动平均耗时和总体平均耗时。准确率必须严格大于 80% 才显示“达标”。

命令行批量验收示例：

```powershell
.\.venv\Scripts\python.exe detailed_action_random_visual_test.py --no-gui --count 500 --seed 20260831 --device cuda
```

每次运行都会在 `outputs/detailed_action_random_visual_test` 下建立独立目录，保存现场生成用例、逐条 JSONL、CSV、汇总 JSON 和 Markdown 报告。单条计时只包围 `DetailedActionParser.resolve()`，不含命令生成、模型初始化、界面刷新和文件写入。
