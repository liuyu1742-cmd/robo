# 家庭任务 Canonical Catalog（15 个验收任务）

来源：`meta/household_task_catalog.json`。

- 验收任务数：15
- cleaning 关联物体数：24
- 全局唯一物体数：124
- 任务-物体关系数：139

## 计数语义

每个物体 ID 在全局唯一物体表中只计一次；同一物体可关联多个任务。共享物体不会重复增加全局物体数，但会形成多条任务-物体关系。

| 验收任务 | 关联物体数 | 物体 |
|---|---:|---|
| `cleaning` | 24 | `carpet`, `wood_floor`, `tile_floor`, `marble_floor`, `laminate_floor`, `vinyl_floor`, `floor_mat`, `robot_vacuum`, `cup`, `mug`, `bowl`, `plate`, `spoon`, `fork`, `knife`, `pot`, `toilet`, `sink`, `bathtub`, `shower`, `mirror`, `bathroom_floor`, `faucet`, `soap` |
| `organizing` | 8 | `storage_box`, `bookshelf`, `wardrobe`, `drawer`, `cabinet`, `desk`, `coffee_table`, `toy` |
| `smart_cooking` | 8 | `rice_cooker`, `induction_cooker`, `oven`, `microwave`, `frying_pan`, `electric_kettle`, `blender`, `pressure_cooker` |
| `appliance_management` | 9 | `ceiling_light`, `reading_lamp`, `television`, `air_conditioner`, `refrigerator`, `washing_machine`, `electric_curtain`, `fan`, `remote_control` |
| `security_monitoring` | 8 | `smart_lock`, `security_camera`, `fire_alarm`, `fire_extinguisher`, `doorknob`, `padlock`, `doorbell_button`, `video_doorbell` |
| `laundry` | 8 | `child_clothing`, `shoes`, `cotton_coat`, `down_jacket`, `wool_garment`, `shirt`, `pants`, `towel` |
| `waste_disposal` | 8 | `trash_bin`, `recyclable_bottle`, `food`, `plastic_bag`, `cardboard_box`, `aluminum_can`, `glass_bottle`, `battery` |
| `clothing_care` | 8 | `dress`, `suit`, `sweater`, `skirt`, `socks`, `scarf`, `bed_sheet`, `iron` |
| `window_care` | 8 | `window`, `window_glass`, `window_frame`, `curtain`, `blind`, `screen_window`, `balcony_railing`, `window_sill` |
| `bedroom_service` | 8 | `bed`, `pillow`, `quilt`, `mattress`, `nightstand`, `bedroom_lamp`, `clothes_hanger`, `laundry_basket` |
| `food_serving` | 8 | `water_cup`, `wine_glass`, `bottle`, `food_tray`, `dinner_plate`, `serving_bowl`, `chopsticks`, `thermos` |
| `object_fetching` | 8 | `remote_control`, `mobile_phone`, `laptop`, `book`, `spectacles`, `walking_cane`, `keys`, `wallet` |
| `elderly_assistance` | 8 | `blood_pressure_monitor`, `thermometer`, `bandage`, `first_aid_kit`, `wheelchair`, `sanitary_pad`, `medical_mask`, `medical_glove` |
| `maintenance_management` | 10 | `robot_vacuum`, `fire_alarm`, `washing_machine`, `air_conditioner`, `refrigerator`, `fan`, `electric_curtain`, `smart_lock`, `flower_pot`, `watering_can` |
| `entertainment_service` | 8 | `remote_control`, `television`, `laptop`, `book`, `mobile_phone`, `toy`, `game_controller`, `speaker` |

## 卧室服务的具体操作

- `bed`：铺平床面并整理床铺
- `pillow`：拍松枕头、摆正朝向并放回床头
- `quilt`：展开并铺平被子，抚平褶皱
- `mattress`：检查并拉直床垫边缘，不抓取整张床垫
- `nightstand`：清空遮挡后摆正床头柜上的物品
- `bedroom_lamp`：打开卧室灯并确认照明状态
- `clothes_hanger`：把衣架挂回衣柜横杆
- `laundry_basket`：把洗衣篮移到指定收纳位置
