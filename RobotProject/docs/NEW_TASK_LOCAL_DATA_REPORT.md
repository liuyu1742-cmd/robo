# 新任务本地数据报告

本报告仅使用 `datasets/midterm_15task_delivery` 内的证据，并由最新轻量 manifest `datasets/new_household_tasks/import_manifest.json` 生成；未搜索或汇总其他 datasets。

- maintenance_management 关联物体：8
- entertainment_service 关联物体：6

共享物体沿用 canonical 全局 ID，只计一次全局唯一物体；关联到多个任务时分别计入任务-物体关系。

| 新任务 | 物体 | 状态 | 图像 | 视频 | 标注 | 证据路径 |
|---|---|---|---:|---:|---:|---|
| `maintenance_management` | `robot_vacuum` | available | 0 | 10 | 10 | `datasets/midterm_15task_delivery/floor_cleaning__地面清洁/objects/robot_vacuum` |
| `maintenance_management` | `fire_alarm` | available | 462 | 0 | 568 | `datasets/midterm_15task_delivery/security_monitoring__智慧安防/objects/fire_alarm` |
| `maintenance_management` | `washing_machine` | available | 466 | 33 | 805 | `datasets/midterm_15task_delivery/appliance_management__家电综合管理/objects/washing_machine` |
| `maintenance_management` | `air_conditioner` | available | 571 | 0 | 1030 | `datasets/midterm_15task_delivery/appliance_management__家电综合管理/objects/air_conditioner` |
| `maintenance_management` | `refrigerator` | available | 513 | 34 | 1099 | `datasets/midterm_15task_delivery/appliance_management__家电综合管理/objects/refrigerator` |
| `maintenance_management` | `fan` | available | 510 | 0 | 1020 | `datasets/midterm_15task_delivery/appliance_management__家电综合管理/objects/fan` |
| `maintenance_management` | `electric_curtain` | available | 300 | 0 | 600 | `datasets/midterm_15task_delivery/appliance_management__家电综合管理/objects/electric_curtain` |
| `maintenance_management` | `smart_lock` | available | 21 | 0 | 42 | `datasets/midterm_15task_delivery/security_monitoring__智慧安防/objects/smart_lock` |
| `entertainment_service` | `remote_control` | available | 2829 | 7 | 4867 | `datasets/midterm_15task_delivery/object_fetching__物品取送/objects/remote_control` |
| `entertainment_service` | `television` | available | 143 | 0 | 308 | `datasets/midterm_15task_delivery/appliance_management__家电综合管理/objects/television` |
| `entertainment_service` | `laptop` | available | 2323 | 0 | 5405 | `datasets/midterm_15task_delivery/object_fetching__物品取送/objects/laptop` |
| `entertainment_service` | `book` | available | 9095 | 19 | 19658 | `datasets/midterm_15task_delivery/object_fetching__物品取送/objects/book` |
| `entertainment_service` | `mobile_phone` | available | 617 | 5 | 2059 | `datasets/midterm_15task_delivery/object_fetching__物品取送/objects/mobile_phone` |
| `entertainment_service` | `toy` | available | 924 | 0 | 2294 | `datasets/midterm_15task_delivery/organizing__整理任务/objects/toy` |
