# 项目目录说明

## VLA 10-task / 82-object action templates

- `tools/vla_action_templates.py`: workbook-derived, read-only template catalogue and deterministic instruction selector.
- `meta/vla_82_detailed_action_sequences.json`: the user-approved 82-row fine-grained action-sequence catalogue.
- `tools/vla_detailed_action_sequences.py`: validates the approved catalogue and resolves it by the existing VLA template ID.
- `manual_instruction_test.py`: applies the selected complete action sequence to the existing parser output.
- Runtime JSON exposes the second sequence as `detailed_action_sequence`; CLI/GUI renders it under `额外细分动作序列`.
- The fine-grained sequence is independent of and does not modify `final_actions`, `expected_actions`, or the original VLA template `steps`.
- The external XLSX, all videos, and all video annotations remain source-only and are never moved or modified by this feature.

## 必须保留

- `models/action_transformer_project_final.pt`：正式动作解析模型。
- `models/coco_household/yolov8n_16cls_small_object_boost_e10/weights/best.pt`：当前视觉检测模型。
- `final_demo.py`：最终演示入口。
- `evaluate_project_model_v2.py`：正式动作模型评估脚本。
- `train_project_transformer_v2.py`：项目版动作模型训练脚本。
- `train_coco_household.py`：视觉检测训练脚本。
- `datasets/project_*_dataset_normalized.json`：项目版训练/验证/测试集。
- `datasets/final_metrics_report.*`：最终指标报告。
- `docs/`：验收说明、覆盖表和演示文档。

## 可重建或阶段性产物

- `models/coco_household/*/predictions.json`：视觉预测结果，可由模型重新导出。
- `models/coco_household/*/threshold_sweep/`：阈值扫描结果，可重新计算。
- `datasets/visual_results/`：可视化图表，可重新生成。
- `datasets/vision_error_analysis*/`：错误样例分析，可重新生成。

## 原始/处理后数据

- `datasets/coco_lvis_balanced_yolo/`：COCO/LVIS 家政视觉检测子集。
- `datasets/hoi4d/`：HOI4D 标注和筛选结果。
- `datasets/hico_20160224_det/`：HICO 相关处理数据。
- `datasets/epic_kitchens/`：EPIC-KITCHENS 标注和筛选结果。

## 归档目录

- `archive/old_scripts/`：旧实验入口和已替代脚本，仅用于历史追溯。
