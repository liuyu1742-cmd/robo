# Medical service 清单替换报告

生成时间：2026-07-13

## 当前口径

- 原 `elderly_assistance` 任务显示名已改为 `Medical service`。
- Medical service 面向家庭健康照护，不再使用偏医院/诊所场景的 `stethoscope`、`syringe`，也不使用与药盒/取药重复的 `medicine_bottle`、`pill_box`。
- 当前 Medical service 的 8 个物体为：
  - `blood_pressure_monitor`
  - `thermometer`
  - `bandage`
  - `first_aid_kit`
  - `wheelchair`
  - `hearing_aid`
  - `medical_mask`
  - `medical_glove`

## 已执行

- `object_fetching` 中 `medicine_box` 已替换为 `spectacles`。
- 原 Medical service 下的 `spectacles` 数据已移动到 `object_fetching / spectacles`。
- `medicine_box` 活跃交付目录已删除，不再进入正式 120 类统计。
- `glucose_meter`、`smart_scale` 已从 Medical service 中移除。
- `stethoscope`、`syringe`、`medicine_bottle`、`pill_box` 已从当前正式清单中移除。

## 后续需要补数据

以下 3 类是新换入的家庭医疗物品，目前需要继续补公开数据集或桌面数据：

- `blood_pressure_monitor`
- `first_aid_kit`
- `hearing_aid`
