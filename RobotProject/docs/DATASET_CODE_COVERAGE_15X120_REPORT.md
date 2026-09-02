# 15+120 数据集与代码证据审计报告

## 结论

- 正式任务数：15
- 正式物体/设备数：124
- 有明确数据集或代码证据：124
- 暂未找到明确数据集或代码证据：0
- 已建立P1本地证据模板：2

这里的“缺”不是指没有动作序列；这些项目仍然有标准指令和标准动作序列。这里缺的是可用于交付说明的公开视觉数据集、映射报告、检测代码或同类证据。

P1本地证据模板表示已经建立采集清单、标注模板和说明文件，但还不能等同于已经采集图片或完成训练。

## 暂缺明确数据集/代码证据的项目

| 任务 | 缺的名字 |
|---|---|

## 已建立P1本地证据模板

`child_clothing`, `video_doorbell`

## 旧覆盖表与正式120清单不一致

正式清单有、旧覆盖表没有：



旧覆盖表有、正式清单没有：



## 证据来源

- `datasets/lvis/processed/lvis_120_coverage_report.json`
- `datasets/open_images_household_51/mapping_report.json`
- `datasets/open_images_household/mapping_report.json`
- `datasets/fashionpedia/project_clothing_mapping.json`
- `datasets/object_operation_coverage_120.json`
- `datasets/midterm_15task_delivery/_delivery_index.csv`

## 建议下一步

1. 先以正式 `meta/compliance_catalog_15x120.json` 为准，重建或同步 `datasets/object_operation_coverage_120.json`。
2. 对13个缺口逐项补充公开数据映射或自采/合成数据说明。
3. 再更新总状态文档，把“清单、动作序列、真实视觉数据证据”三类状态分开写。
