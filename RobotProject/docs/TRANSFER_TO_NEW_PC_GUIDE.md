# RobotProject 迁移到新电脑指南

## 1. 推荐迁移方式

建议打包项目源码、模型、数据和文档，但不要打包虚拟环境、Git 内部目录和缓存。

当前项目主体约 74GB，主要体积来自 `datasets/`。`.venv` 约 4.5GB，不建议迁移；新电脑上应重新配置 conda 环境。

## 2. 推荐打包内容

必须包含：

```text
configs/
datasets/
docs/
meta/
models/
tools/
archive/
evaluate_project_actions.py
evaluate_project_model_v2.py
final_demo.py
run_acceptance_check.py
train_coco_household.py
train_project_transformer_v2.py
PROJECT_DELIVERY_SUMMARY.md
PYCHARM_RUN_GUIDE.md
VISION_NEXT_ROUND_PLAN.md
requirements.txt
requirements-cuda.txt
requirements-robocasa311-freeze.txt
environment-robocasa311.yml
```

不要打包：

```text
.venv/
.git/
.agents/
.codex/
.idea/
__pycache__/
tools/__pycache__/
*.pyc
```

## 3. 完整包 vs 轻量包

### 方案 A：完整包，推荐

适合另一台电脑要继续当前工作，包括视觉训练、HOI4D/EPIC 处理、COCO/LVIS 复用。

包含：

```text
datasets/
models/
tools/
configs/
meta/
docs/
根目录正式脚本
```

优点：新电脑基本不用重新下载已有数据。

缺点：体积大，压缩后仍可能在 55GB～75GB 左右，因为图片、模型和部分数据本来就是压缩格式。

### 方案 B：轻量包

适合只看代码、报告、模型演示，不继续视觉训练。

可排除：

```text
datasets/coco/
datasets/hico_20160224_det/images/
datasets/coco_lvis_balanced_yolo/images/
datasets/coco_lvis_balanced_yolo/labels/
```

优点：体积会小很多。

缺点：不能直接继续视觉训练，也不能完整复现实验。

当前你说要在另一台电脑处理后续工作，所以推荐方案 A。

## 4. 使用 7-Zip 打包建议

如果安装了 7-Zip，可在 PowerShell 中运行类似命令：

```powershell
cd D:\
7z a -t7z RobotProject_transfer.7z RobotProject\ -xr!RobotProject\.venv -xr!RobotProject\.git -xr!RobotProject\.agents -xr!RobotProject\.codex -xr!RobotProject\.idea -xr!RobotProject\__pycache__ -xr!RobotProject\tools\__pycache__ -xr!*.pyc
```

如果不使用命令行，也可以右键压缩 `D:\RobotProject`，手动排除上面的目录。

## 5. 新电脑恢复步骤

### 5.1 解压目录

建议仍放到：

```text
D:\RobotProject
```

如果放到别的路径，部分 YAML 里有绝对路径，需要改：

```text
datasets/coco_lvis_balanced_yolo/household_small_object_boost.yaml
datasets/coco_lvis_balanced_yolo/household_compact_balanced.yaml
datasets/coco_lvis_balanced_yolo/household_balanced.yaml
```

这些文件里的：

```text
path: D:/RobotProject/datasets/coco_lvis_balanced_yolo
```

需要改成新电脑上的实际路径。

### 5.2 创建环境

优先用 conda：

```powershell
conda env create -f environment-robocasa311.yml
conda activate robocasa311
```

如果环境创建失败，可手动创建 Python 3.11 环境：

```powershell
conda create -n robocasa311 python=3.11 -y
conda activate robocasa311
python -m pip install -r requirements-robocasa311-freeze.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果需要 GPU 版 PyTorch，优先按新电脑 CUDA/显卡情况安装；原项目当前使用过：

```powershell
python -m pip install --upgrade torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

国内普通 Python 包优先用清华源：

```powershell
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.3 PyCharm 设置

1. 打开 `D:\RobotProject`；
2. 设置解释器为新环境的 Python，例如：

```text
D:\anaconda3\envs\robocasa311\python.exe
```

3. 在 PyCharm 终端运行自检：

```powershell
D:\anaconda3\envs\robocasa311\python.exe run_acceptance_check.py
```

4. 再运行最终演示：

```powershell
D:\anaconda3\envs\robocasa311\python.exe final_demo.py --seed 7
```

## 6. 迁移后优先检查

确认这些文件存在：

```text
models/action_transformer_project_final.pt
models/coco_household/yolov8n_16cls_small_object_boost_e10/weights/best.pt
datasets/project_test_dataset_normalized.json
datasets/object_operation_coverage_120.json
docs/DATASET_STATUS_REPORT.md
docs/HOI4D_DOWNLOAD_PLAN.md
```

确认这些命令能跑：

```powershell
python run_acceptance_check.py
python final_demo.py --seed 7
python tools\build_dataset_status_report.py
```

## 7. 迁移后下一阶段工作

新电脑上建议继续：

1. 按 `docs/HOI4D_DOWNLOAD_PLAN.md` 下载 HOI4D RGB Video、Hand Pose、Camera Parameters；
2. 整理到 `datasets/hoi4d/raw/`；
3. 编写/运行 HOI4D 视觉索引脚本；
4. 按 EPIC 已筛选清单下载精选视频；
5. 扩展 COCO/LVIS 到 30～40 个家政视觉类别。
