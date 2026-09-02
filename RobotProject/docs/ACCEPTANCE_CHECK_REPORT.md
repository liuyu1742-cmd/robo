# 一键验收检查报告

- 总体状态：未通过
- 缺失文件数：0

## 核心指标

- 冻结指令整序列准确率：100.00%
- 冻结指令步骤准确率：100.00%
- 指令字段解析准确率：100.00%
- 冻结测试样例数：139
- 冻结基准任务/物体覆盖：15/124
- 指令模板跨集合重叠数：0
- 指定90%要求：未通过
- 验收任务数：15（要求恰好 15）
- cleaning 关联物体数：23（要求 24）
- 全局唯一物体数：123（要求至少 124）
- 任务-物体关系数：138（当前要求 139）
- maintenance/entertainment 关联数：10/8（要求 8/6）
- 旧 cleaning ID 映射：通过
- 轻量引用目录/行：3/42（要求 3/38）
- 引用行图片/视频/标注数：0/0/0
- 本地唯一图片/视频/标注总数：99117/3020/186923
- 本地数据来源：datasets/midterm_15task_delivery
- 引用媒体复制数：0
- 期中目录外引用数：4
- split 泄漏检查适用：False（Reference-only views contain no train/val/test split metadata.）
- 任务层级检查：未通过
- 唯一物体范围检查：未通过
- 本地数据完整性检查：未通过
- 旧教师强制动作序列准确率（参考）：99.37%
- 旧动作步骤覆盖率（参考）：99.76%
- 仅视频动词准确率：30.79%
- 仅视频任务准确率：77.22%
- 仅视频物体准确率：8.96%
- 仅视频联合准确率：2.00%
- 视频联合90%（扩展研究，不作为本轮门槛）：未达到
- 120 物体覆盖：123
- 15 类任务演示样例：15
- 视觉直接检测覆盖对象数：0
- 视觉最优 F1：49.86%
- 推荐视觉阈值：conf=0.3
- 视觉错误图库图片数：18

## 文件检查

| 项目 | 状态 | 路径 |
|---|---|---|
| 正式验收指令基准 | 存在 | `datasets\acceptance_instruction_action_benchmark_15tasks.json` |
| 指令动作模型 | 存在 | `models\instruction_action_transformer_v1.pt` |
| 指令动作正式报告 | 存在 | `datasets\instruction_action_benchmark_report.json` |
| 正式动作模型 | 存在 | `models\action_transformer_project_final.pt` |
| 当前视觉模型 | 存在 | `models\coco_household\yolov8n_16cls_small_object_boost_e10\weights\best.pt` |
| 动作模型评估报告 | 存在 | `datasets\project_model_final_normalized_eval_report.json` |
| 最终指标 JSON | 存在 | `datasets\final_metrics_report.json` |
| 最终指标 Markdown | 存在 | `datasets\final_metrics_report.md` |
| 120 物体覆盖 JSON | 存在 | `datasets\object_operation_coverage_120.json` |
| 120 物体覆盖文档 | 存在 | `docs\120_OBJECT_OPERATION_COVERAGE.md` |
| 15 任务演示 JSON | 存在 | `datasets\final_demo_cases_15tasks.json` |
| 15 任务演示文档 | 存在 | `docs\FINAL_DEMO_CASES.md` |
| 项目结构说明 | 存在 | `docs\PROJECT_STRUCTURE.md` |
| 总交付说明 | 存在 | `PROJECT_DELIVERY_SUMMARY.md` |
| PyCharm 运行指南 | 存在 | `PYCHARM_RUN_GUIDE.md` |
| 最终演示入口 | 存在 | `final_demo.py` |
| 视觉错误图库 | 存在 | `datasets\visual_error_gallery\visual_error_gallery.md` |
| 动作指标图 | 存在 | `datasets\visual_results\action_metrics.png` |
| 视觉总体指标图 | 存在 | `datasets\visual_results\vision_overall_metrics.png` |
| 视觉弱类图 | 存在 | `datasets\visual_results\vision_weak_classes.png` |
| 家政任务 canonical catalog | 存在 | `meta\household_task_catalog.json` |
| 期中交付索引 | 存在 | `datasets\midterm_15task_delivery\_delivery_index.csv` |
| 验收任务目录说明 | 存在 | `datasets\midterm_15task_delivery\README_ACCEPTANCE_TASKS.md` |