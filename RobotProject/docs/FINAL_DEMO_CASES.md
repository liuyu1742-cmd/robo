# 15 个验收任务最终演示样例

每个 acceptance task 至少提供一个代表动作案例。cleaning 样例保留旧清洁子任务 `task`，同时标记 `acceptance_task=cleaning`。

| 序号 | 验收任务 | 兼容 task | 指令 | 对象 | 动作序列 |
|---:|---|---|---|---|---|
| 1 | `cleaning` | `floor_cleaning` | 清洁 carpet | `carpet` | `locate(carpet) → clean(carpet) → inspect(carpet)` |
| 2 | `organizing` | `organizing` | 执行 organizing：操作 storage_box | `storage_box` | `locate(storage_box) → grasp(storage_box) → move(storage) → place(storage_box)` |
| 3 | `smart_cooking` | `smart_cooking` | 执行 smart_cooking：操作 rice_cooker | `rice_cooker` | `locate(rice_cooker) → turn_on(rice_cooker) → cook(rice_cooker) → turn_off(rice_cooker)` |
| 4 | `appliance_management` | `appliance_management` | 执行 appliance_management：操作 ceiling_light | `ceiling_light` | `locate(ceiling_light) → turn_on(ceiling_light) → inspect(ceiling_light)` |
| 5 | `security_monitoring` | `security_monitoring` | 执行 security_monitoring：操作 smart_lock | `smart_lock` | `locate(smart_lock) → lock(smart_lock) → inspect(smart_lock)` |
| 6 | `laundry` | `laundry` | 执行 laundry：操作 child_clothing | `child_clothing` | `locate(child_clothing) → grasp(child_clothing) → move(washer) → wash(child_clothing) → dry(child_clothing)` |
| 7 | `waste_disposal` | `waste_disposal` | 执行 waste_disposal：操作 trash_bin | `trash_bin` | `locate(trash_bin) → grasp(trash_bin) → move(trash_bin) → dispose(trash_bin)` |
| 8 | `clothing_care` | `clothing_care` | 执行 clothing_care：操作 dress | `dress` | `locate(dress) → grasp(dress) → fold(dress) → move(wardrobe) → place(dress)` |
| 9 | `window_care` | `window_care` | 执行 window_care：操作 window | `window` | `locate(window) → open(window) → inspect(window)` |
| 10 | `bedroom_service` | `bedroom_service` | 铺平床面并整理床铺 | `bed` | `locate(bed) → straighten(bed) → inspect(bed)` |
| 11 | `food_serving` | `food_serving` | 执行 food_serving：操作 water_cup | `water_cup` | `locate(water_cup) → grasp(water_cup) → move(water_source) → fill(water_cup) → pour(water_cup) → move(person) → place(water_cup)` |
| 12 | `object_fetching` | `object_fetching` | 执行 object_fetching：操作 remote_control | `remote_control` | `locate(remote_control) → grasp(remote_control) → move(person) → release(remote_control)` |
| 13 | `elderly_assistance` | `elderly_assistance` | 执行 elderly_assistance：操作 blood_pressure_monitor | `blood_pressure_monitor` | `locate(blood_pressure_monitor) → grasp(blood_pressure_monitor) → move(person) → place(blood_pressure_monitor) → inspect(blood_pressure_monitor)` |
| 14 | `maintenance_management` | `maintenance_management` | 检查并养护 robot_vacuum | `robot_vacuum` | `locate(robot_vacuum) → inspect(robot_vacuum) → service_brush(robot_vacuum) → maintain(robot_vacuum) → confirm_state(robot_vacuum)` |
| 15 | `entertainment_service` | `entertainment_service` | 使用 remote_control 提供娱乐服务 | `remote_control` | `locate(remote_control) → grasp(remote_control) → aim(remote_control,television) → press(remote_control) → control(remote_control,television) → confirm_state(television)` |
