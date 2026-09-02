# 家庭任务物体操作覆盖表

来源：`meta/household_task_catalog.json`（canonical catalog）。

- 验收任务数：15
- cleaning 关联物体数：23
- 全局唯一物体数：123
- 任务-物体关系数：138

物体在全局目录中只计一次，但同一物体可以关联多个任务，因此关系数可以大于唯一物体数。

## 清洁 `cleaning`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `floor_cleaning` | `carpet` | 清洁 carpet | `locate(carpet) → clean(carpet) → inspect(carpet)` |
| 2 | `floor_cleaning` | `wood_floor` | 清洁 wood_floor | `locate(wood_floor) → clean(wood_floor) → inspect(wood_floor)` |
| 3 | `floor_cleaning` | `tile_floor` | 清洁 tile_floor | `locate(tile_floor) → clean(tile_floor) → inspect(tile_floor)` |
| 4 | `floor_cleaning` | `marble_floor` | 清洁 marble_floor | `locate(marble_floor) → clean(marble_floor) → inspect(marble_floor)` |
| 5 | `floor_cleaning` | `laminate_floor` | 清洁 laminate_floor | `locate(laminate_floor) → clean(laminate_floor) → inspect(laminate_floor)` |
| 6 | `floor_cleaning` | `vinyl_floor` | 清洁 vinyl_floor | `locate(vinyl_floor) → clean(vinyl_floor) → inspect(vinyl_floor)` |
| 7 | `floor_cleaning` | `robot_vacuum` | 清洁 robot_vacuum | `locate(robot_vacuum) → turn_on(robot_vacuum) → clean(robot_vacuum) → inspect(robot_vacuum) → turn_off(robot_vacuum)` |
| 8 | `dish_washing` | `cup` | 清洁 cup | `locate(cup) → grasp(cup) → move(sink) → wash(cup) → rinse(cup) → place(cup)` |
| 9 | `dish_washing` | `mug` | 清洁 mug | `locate(mug) → grasp(mug) → move(sink) → wash(mug) → rinse(mug) → place(mug)` |
| 10 | `dish_washing` | `bowl` | 清洁 bowl | `locate(bowl) → grasp(bowl) → move(sink) → wash(bowl) → rinse(bowl) → place(bowl)` |
| 11 | `dish_washing` | `plate` | 清洁 plate | `locate(plate) → grasp(plate) → move(sink) → wash(plate) → rinse(plate) → place(plate)` |
| 12 | `dish_washing` | `spoon` | 清洁 spoon | `locate(spoon) → grasp(spoon) → move(sink) → wash(spoon) → rinse(spoon) → place(spoon)` |
| 13 | `dish_washing` | `fork` | 清洁 fork | `locate(fork) → grasp(fork) → move(sink) → wash(fork) → rinse(fork) → place(fork)` |
| 14 | `dish_washing` | `knife` | 清洁 knife | `locate(knife) → grasp(knife) → move(sink) → wash(knife) → rinse(knife) → place(knife)` |
| 15 | `dish_washing` | `pot` | 清洁 pot | `locate(pot) → grasp(pot) → move(sink) → wash(pot) → rinse(pot) → place(pot)` |
| 16 | `bathroom_cleaning` | `toilet` | 清洁 toilet | `locate(toilet) → scrub(toilet) → rinse(toilet) → inspect(toilet)` |
| 17 | `bathroom_cleaning` | `sink` | 清洁 sink | `locate(sink) → scrub(sink) → rinse(sink) → inspect(sink)` |
| 18 | `bathroom_cleaning` | `bathtub` | 清洁 bathtub | `locate(bathtub) → scrub(bathtub) → rinse(bathtub) → inspect(bathtub)` |
| 19 | `bathroom_cleaning` | `shower` | 清洁 shower | `locate(shower) → scrub(shower) → rinse(shower) → inspect(shower)` |
| 20 | `bathroom_cleaning` | `mirror` | 清洁 mirror | `locate(mirror) → scrub(mirror) → rinse(mirror) → inspect(mirror)` |
| 21 | `bathroom_cleaning` | `bathroom_floor` | 清洁 bathroom_floor | `locate(bathroom_floor) → scrub(bathroom_floor) → rinse(bathroom_floor) → inspect(bathroom_floor)` |
| 22 | `bathroom_cleaning` | `faucet` | 清洁 faucet | `locate(faucet) → scrub(faucet) → rinse(faucet) → inspect(faucet)` |
| 23 | `bathroom_cleaning` | `soap` | 清洁 soap | `locate(soap) → scrub(soap) → rinse(soap) → inspect(soap)` |

## 整理任务 `organizing`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `organizing` | `storage_box` | 执行 organizing：操作 storage_box | `locate(storage_box) → grasp(storage_box) → move(storage) → place(storage_box)` |
| 2 | `organizing` | `bookshelf` | 执行 organizing：操作 bookshelf | `locate(bookshelf) → grasp(bookshelf) → move(storage) → place(bookshelf)` |
| 3 | `organizing` | `wardrobe` | 执行 organizing：操作 wardrobe | `locate(wardrobe) → grasp(wardrobe) → move(storage) → place(wardrobe)` |
| 4 | `organizing` | `drawer` | 执行 organizing：操作 drawer | `locate(drawer) → grasp(drawer) → move(storage) → place(drawer)` |
| 5 | `organizing` | `cabinet` | 执行 organizing：操作 cabinet | `locate(cabinet) → grasp(cabinet) → move(storage) → place(cabinet)` |
| 6 | `organizing` | `desk` | 执行 organizing：操作 desk | `locate(desk) → grasp(desk) → move(storage) → place(desk)` |
| 7 | `organizing` | `coffee_table` | 执行 organizing：操作 coffee_table | `locate(coffee_table) → grasp(coffee_table) → move(storage) → place(coffee_table)` |
| 8 | `organizing` | `toy` | 执行 organizing：操作 toy | `locate(toy) → grasp(toy) → move(storage) → place(toy)` |

## 智能烹饪 `smart_cooking`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `smart_cooking` | `rice_cooker` | 执行 smart_cooking：操作 rice_cooker | `locate(rice_cooker) → turn_on(rice_cooker) → cook(rice_cooker) → turn_off(rice_cooker)` |
| 2 | `smart_cooking` | `induction_cooker` | 执行 smart_cooking：操作 induction_cooker | `locate(induction_cooker) → turn_on(induction_cooker) → cook(induction_cooker) → turn_off(induction_cooker)` |
| 3 | `smart_cooking` | `oven` | 执行 smart_cooking：操作 oven | `locate(oven) → turn_on(oven) → cook(oven) → turn_off(oven)` |
| 4 | `smart_cooking` | `microwave` | 执行 smart_cooking：操作 microwave | `locate(microwave) → turn_on(microwave) → cook(microwave) → turn_off(microwave)` |
| 5 | `smart_cooking` | `frying_pan` | 执行 smart_cooking：操作 frying_pan | `locate(frying_pan) → turn_on(frying_pan) → cook(frying_pan) → turn_off(frying_pan)` |
| 6 | `smart_cooking` | `electric_kettle` | 执行 smart_cooking：操作 electric_kettle | `locate(electric_kettle) → turn_on(electric_kettle) → cook(electric_kettle) → turn_off(electric_kettle)` |
| 7 | `smart_cooking` | `blender` | 执行 smart_cooking：操作 blender | `locate(blender) → turn_on(blender) → cook(blender) → turn_off(blender)` |
| 8 | `smart_cooking` | `pressure_cooker` | 执行 smart_cooking：操作 pressure_cooker | `locate(pressure_cooker) → turn_on(pressure_cooker) → cook(pressure_cooker) → turn_off(pressure_cooker)` |

## 家电综合管理 `appliance_management`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `appliance_management` | `ceiling_light` | 执行 appliance_management：操作 ceiling_light | `locate(ceiling_light) → turn_on(ceiling_light) → inspect(ceiling_light)` |
| 2 | `appliance_management` | `reading_lamp` | 执行 appliance_management：操作 reading_lamp | `locate(reading_lamp) → turn_on(reading_lamp) → inspect(reading_lamp)` |
| 3 | `appliance_management` | `television` | 执行 appliance_management：操作 television | `locate(television) → turn_on(television) → inspect(television)` |
| 4 | `appliance_management` | `air_conditioner` | 执行 appliance_management：操作 air_conditioner | `locate(air_conditioner) → turn_on(air_conditioner) → inspect(air_conditioner)` |
| 5 | `appliance_management` | `refrigerator` | 执行 appliance_management：操作 refrigerator | `locate(refrigerator) → turn_off(refrigerator) → inspect(refrigerator)` |
| 6 | `appliance_management` | `washing_machine` | 执行 appliance_management：操作 washing_machine | `locate(washing_machine) → turn_on(washing_machine) → inspect(washing_machine)` |
| 7 | `appliance_management` | `electric_curtain` | 执行 appliance_management：操作 electric_curtain | `locate(electric_curtain) → turn_on(electric_curtain) → inspect(electric_curtain)` |
| 8 | `appliance_management` | `fan` | 执行 appliance_management：操作 fan | `locate(fan) → turn_on(fan) → inspect(fan)` |
| 9 | `appliance_management` | `remote_control` | 执行 appliance_management：操作 remote_control | `locate(remote_control) → turn_on(remote_control) → inspect(remote_control)` |

## 智慧安防 `security_monitoring`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `security_monitoring` | `smart_lock` | 执行 security_monitoring：操作 smart_lock | `locate(smart_lock) → lock(smart_lock) → inspect(smart_lock)` |
| 2 | `security_monitoring` | `security_camera` | 执行 security_monitoring：操作 security_camera | `locate(security_camera) → inspect(security_camera)` |
| 3 | `security_monitoring` | `fire_alarm` | 执行 security_monitoring：操作 fire_alarm | `locate(fire_alarm) → inspect(fire_alarm)` |
| 4 | `security_monitoring` | `fire_extinguisher` | 执行 security_monitoring：操作 fire_extinguisher | `locate(fire_extinguisher) → inspect(fire_extinguisher)` |
| 5 | `security_monitoring` | `doorknob` | 执行 security_monitoring：操作 doorknob | `locate(doorknob) → inspect(doorknob)` |
| 6 | `security_monitoring` | `padlock` | 执行 security_monitoring：操作 padlock | `locate(padlock) → lock(padlock) → inspect(padlock)` |
| 7 | `security_monitoring` | `doorbell_button` | 执行 security_monitoring：操作 doorbell_button | `locate(doorbell_button) → press(doorbell_button) → inspect(doorbell_button)` |
| 8 | `security_monitoring` | `video_doorbell` | 执行 security_monitoring：操作 video_doorbell | `locate(video_doorbell) → inspect(video_doorbell)` |

## 衣物清洗 `laundry`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `laundry` | `child_clothing` | 执行 laundry：操作 child_clothing | `locate(child_clothing) → grasp(child_clothing) → move(washer) → wash(child_clothing) → dry(child_clothing) → inspect(child_clothing)` |
| 2 | `laundry` | `shoes` | 执行 laundry：操作 shoes | `locate(shoes) → grasp(shoes) → move(washer) → wash(shoes) → dry(shoes) → inspect(shoes)` |
| 3 | `laundry` | `cotton_coat` | 执行 laundry：操作 cotton_coat | `locate(cotton_coat) → grasp(cotton_coat) → move(washer) → wash(cotton_coat) → dry(cotton_coat) → inspect(cotton_coat)` |
| 4 | `laundry` | `down_jacket` | 执行 laundry：操作 down_jacket | `locate(down_jacket) → grasp(down_jacket) → move(washer) → wash(down_jacket) → dry(down_jacket) → inspect(down_jacket)` |
| 5 | `laundry` | `wool_garment` | 执行 laundry：操作 wool_garment | `locate(wool_garment) → grasp(wool_garment) → move(washer) → wash(wool_garment) → dry(wool_garment) → inspect(wool_garment)` |
| 6 | `laundry` | `shirt` | 执行 laundry：操作 shirt | `locate(shirt) → grasp(shirt) → move(washer) → wash(shirt) → dry(shirt) → inspect(shirt)` |
| 7 | `laundry` | `pants` | 执行 laundry：操作 pants | `locate(pants) → grasp(pants) → move(washer) → wash(pants) → dry(pants) → inspect(pants)` |
| 8 | `laundry` | `towel` | 执行 laundry：操作 towel | `locate(towel) → grasp(towel) → move(washer) → wash(towel) → dry(towel) → inspect(towel)` |

## 垃圾处理 `waste_disposal`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `waste_disposal` | `trash_bin` | 执行 waste_disposal：操作 trash_bin | `locate(trash_bin) → grasp(trash_bin) → move(trash_bin) → dispose(trash_bin)` |
| 2 | `waste_disposal` | `recyclable_bottle` | 执行 waste_disposal：操作 recyclable_bottle | `locate(recyclable_bottle) → grasp(recyclable_bottle) → move(trash_bin) → dispose(recyclable_bottle)` |
| 3 | `waste_disposal` | `food` | 执行 waste_disposal：操作 food | `locate(food) → grasp(food) → move(trash_bin) → dispose(food)` |
| 4 | `waste_disposal` | `plastic_bag` | 执行 waste_disposal：操作 plastic_bag | `locate(plastic_bag) → grasp(plastic_bag) → move(trash_bin) → dispose(plastic_bag)` |
| 5 | `waste_disposal` | `cardboard_box` | 执行 waste_disposal：操作 cardboard_box | `locate(cardboard_box) → grasp(cardboard_box) → move(trash_bin) → dispose(cardboard_box)` |
| 6 | `waste_disposal` | `aluminum_can` | 执行 waste_disposal：操作 aluminum_can | `locate(aluminum_can) → grasp(aluminum_can) → move(trash_bin) → dispose(aluminum_can)` |
| 7 | `waste_disposal` | `glass_bottle` | 执行 waste_disposal：操作 glass_bottle | `locate(glass_bottle) → grasp(glass_bottle) → move(trash_bin) → dispose(glass_bottle)` |
| 8 | `waste_disposal` | `battery` | 执行 waste_disposal：操作 battery | `locate(battery) → grasp(battery) → move(trash_bin) → dispose(battery)` |

## 服装护理 `clothing_care`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `clothing_care` | `dress` | 执行 clothing_care：操作 dress | `locate(dress) → grasp(dress) → fold(dress) → move(wardrobe) → place(dress)` |
| 2 | `clothing_care` | `suit` | 执行 clothing_care：操作 suit | `locate(suit) → grasp(suit) → fold(suit) → move(wardrobe) → place(suit)` |
| 3 | `clothing_care` | `sweater` | 执行 clothing_care：操作 sweater | `locate(sweater) → grasp(sweater) → fold(sweater) → move(wardrobe) → place(sweater)` |
| 4 | `clothing_care` | `skirt` | 执行 clothing_care：操作 skirt | `locate(skirt) → grasp(skirt) → fold(skirt) → move(wardrobe) → place(skirt)` |
| 5 | `clothing_care` | `socks` | 执行 clothing_care：操作 socks | `locate(socks) → grasp(socks) → fold(socks) → move(wardrobe) → place(socks)` |
| 6 | `clothing_care` | `scarf` | 执行 clothing_care：操作 scarf | `locate(scarf) → grasp(scarf) → fold(scarf) → move(wardrobe) → place(scarf)` |
| 7 | `clothing_care` | `bed_sheet` | 执行 clothing_care：操作 bed_sheet | `locate(bed_sheet) → grasp(iron) → iron(bed_sheet) → inspect(bed_sheet)` |
| 8 | `clothing_care` | `iron` | 执行 clothing_care：操作 iron | `locate(iron) → inspect(iron)` |

## 门窗护理 `window_care`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `window_care` | `window` | 执行 window_care：操作 window | `locate(window) → open(window) → inspect(window)` |
| 2 | `window_care` | `window_glass` | 执行 window_care：操作 window_glass | `locate(window_glass) → wipe(window_glass) → inspect(window_glass)` |
| 3 | `window_care` | `window_frame` | 执行 window_care：操作 window_frame | `locate(window_frame) → wipe(window_frame) → inspect(window_frame)` |
| 4 | `window_care` | `curtain` | 执行 window_care：操作 curtain | `locate(curtain) → open(curtain) → inspect(curtain)` |
| 5 | `window_care` | `blind` | 执行 window_care：操作 blind | `locate(blind) → open(blind) → inspect(blind)` |
| 6 | `window_care` | `screen_window` | 执行 window_care：操作 screen_window | `locate(screen_window) → wipe(screen_window) → inspect(screen_window)` |
| 7 | `window_care` | `balcony_railing` | 执行 window_care：操作 balcony_railing | `locate(balcony_railing) → wipe(balcony_railing) → inspect(balcony_railing)` |
| 8 | `window_care` | `window_sill` | 执行 window_care：操作 window_sill | `locate(window_sill) → wipe(window_sill) → inspect(window_sill)` |

## 卧室服务 `bedroom_service`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `bedroom_service` | `bed` | 铺平床面并整理床铺 | `locate(bed) → straighten(bed) → inspect(bed)` |
| 2 | `bedroom_service` | `pillow` | 拍松枕头、摆正朝向并放回床头 | `locate(pillow) → grasp(pillow) → fluff(pillow) → orient(pillow) → place(pillow) → inspect(pillow)` |
| 3 | `bedroom_service` | `quilt` | 展开并铺平被子，抚平褶皱 | `locate(quilt) → grasp(quilt) → spread(quilt) → smooth(quilt) → inspect(quilt)` |
| 4 | `bedroom_service` | `mattress` | 检查并拉直床垫边缘，不抓取整张床垫 | `locate(mattress) → inspect(mattress) → straighten(mattress) → inspect(mattress)` |
| 5 | `bedroom_service` | `nightstand` | 清空遮挡后摆正床头柜上的物品 | `locate(nightstand) → organize(nightstand) → wipe(nightstand) → inspect(nightstand)` |
| 6 | `bedroom_service` | `bedroom_lamp` | 打开卧室灯并确认照明状态 | `locate(bedroom_lamp) → turn_on(bedroom_lamp) → inspect(bedroom_lamp)` |
| 7 | `bedroom_service` | `clothes_hanger` | 把衣架挂回衣柜横杆 | `locate(clothes_hanger) → grasp(clothes_hanger) → orient(clothes_hanger) → place(clothes_hanger) → inspect(clothes_hanger)` |
| 8 | `bedroom_service` | `laundry_basket` | 把洗衣篮移到指定收纳位置 | `locate(laundry_basket) → organize(laundry_basket) → move(laundry_basket) → place(laundry_basket) → inspect(laundry_basket)` |

## 送餐饮水 `food_serving`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `food_serving` | `water_cup` | 执行 food_serving：操作 water_cup | `locate(water_cup) → grasp(water_cup) → move(water_source) → fill(water_cup) → pour(water_cup) → move(person) → place(water_cup)` |
| 2 | `food_serving` | `wine_glass` | 执行 food_serving：操作 wine_glass | `locate(wine_glass) → grasp(wine_glass) → move(person) → place(wine_glass)` |
| 3 | `food_serving` | `bottle` | 执行 food_serving：操作 bottle | `locate(bottle) → grasp(bottle) → move(person) → place(bottle)` |
| 4 | `food_serving` | `food_tray` | 执行 food_serving：操作 food_tray | `locate(food_tray) → grasp(food_tray) → move(person) → place(food_tray)` |
| 5 | `food_serving` | `dinner_plate` | 执行 food_serving：操作 dinner_plate | `locate(dinner_plate) → grasp(dinner_plate) → move(person) → place(dinner_plate)` |
| 6 | `food_serving` | `serving_bowl` | 执行 food_serving：操作 serving_bowl | `locate(serving_bowl) → grasp(serving_bowl) → move(person) → place(serving_bowl)` |
| 7 | `food_serving` | `chopsticks` | 执行 food_serving：操作 chopsticks | `locate(chopsticks) → grasp(chopsticks) → move(person) → place(chopsticks)` |
| 8 | `food_serving` | `thermos` | 执行 food_serving：操作 thermos | `locate(thermos) → grasp(thermos) → move(person) → place(thermos)` |

## 物品取送 `object_fetching`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `object_fetching` | `remote_control` | 执行 object_fetching：操作 remote_control | `locate(remote_control) → grasp(remote_control) → move(person) → release(remote_control)` |
| 2 | `object_fetching` | `mobile_phone` | 执行 object_fetching：操作 mobile_phone | `locate(mobile_phone) → grasp(mobile_phone) → move(person) → release(mobile_phone)` |
| 3 | `object_fetching` | `laptop` | 执行 object_fetching：操作 laptop | `locate(laptop) → grasp(laptop) → move(person) → release(laptop)` |
| 4 | `object_fetching` | `book` | 执行 object_fetching：操作 book | `locate(book) → grasp(book) → move(person) → release(book)` |
| 5 | `object_fetching` | `spectacles` | 执行 object_fetching：操作 spectacles | `locate(spectacles) → grasp(spectacles) → move(person) → release(spectacles)` |
| 6 | `object_fetching` | `walking_cane` | 执行 object_fetching：操作 walking_cane | `locate(walking_cane) → grasp(walking_cane) → move(person) → release(walking_cane)` |
| 7 | `object_fetching` | `keys` | 执行 object_fetching：操作 keys | `locate(keys) → grasp(keys) → move(person) → release(keys)` |
| 8 | `object_fetching` | `wallet` | 执行 object_fetching：操作 wallet | `locate(wallet) → grasp(wallet) → move(person) → release(wallet)` |

## 医疗服务 `elderly_assistance`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `elderly_assistance` | `blood_pressure_monitor` | 执行 elderly_assistance：操作 blood_pressure_monitor | `locate(blood_pressure_monitor) → inspect(blood_pressure_monitor)` |
| 2 | `elderly_assistance` | `thermometer` | 执行 elderly_assistance：操作 thermometer | `locate(thermometer) → inspect(thermometer)` |
| 3 | `elderly_assistance` | `bandage` | 执行 elderly_assistance：操作 bandage | `locate(bandage) → grasp(bandage) → move(person) → place(bandage) → inspect(bandage)` |
| 4 | `elderly_assistance` | `first_aid_kit` | 执行 elderly_assistance：操作 first_aid_kit | `locate(first_aid_kit) → grasp(first_aid_kit) → move(person) → place(first_aid_kit) → inspect(first_aid_kit)` |
| 5 | `elderly_assistance` | `wheelchair` | 执行 elderly_assistance：操作 wheelchair | `locate(wheelchair) → grasp(wheelchair) → move(person) → place(wheelchair) → inspect(wheelchair)` |
| 6 | `elderly_assistance` | `sanitary_pad` | 执行 elderly_assistance：操作 sanitary_pad | `locate(sanitary_pad) → grasp(sanitary_pad) → move(person) → place(sanitary_pad) → inspect(sanitary_pad)` |
| 7 | `elderly_assistance` | `medical_mask` | 执行 elderly_assistance：操作 medical_mask | `locate(medical_mask) → grasp(medical_mask) → move(person) → place(medical_mask) → inspect(medical_mask)` |
| 8 | `elderly_assistance` | `medical_glove` | 执行 elderly_assistance：操作 medical_glove | `locate(medical_glove) → grasp(medical_glove) → move(person) → place(medical_glove) → inspect(medical_glove)` |

## 维护管理 `maintenance_management`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `maintenance_management` | `robot_vacuum` | 检查并养护 robot_vacuum | `locate(robot_vacuum) → inspect(robot_vacuum) → service_brush(robot_vacuum) → maintain(robot_vacuum) → confirm_state(robot_vacuum)` |
| 2 | `maintenance_management` | `fire_alarm` | 检查并养护 fire_alarm | `locate(fire_alarm) → inspect(fire_alarm) → test_alarm(fire_alarm) → maintain(fire_alarm) → confirm_state(fire_alarm)` |
| 3 | `maintenance_management` | `washing_machine` | 检查并养护 washing_machine | `locate(washing_machine) → inspect(washing_machine) → clean_drum(washing_machine) → maintain(washing_machine) → confirm_state(washing_machine)` |
| 4 | `maintenance_management` | `air_conditioner` | 检查并养护 air_conditioner | `locate(air_conditioner) → inspect(air_conditioner) → clean_filter(air_conditioner) → maintain(air_conditioner) → confirm_state(air_conditioner)` |
| 5 | `maintenance_management` | `refrigerator` | 检查并养护 refrigerator | `locate(refrigerator) → inspect(refrigerator) → check_seal(refrigerator) → maintain(refrigerator) → confirm_state(refrigerator)` |
| 6 | `maintenance_management` | `fan` | 检查并养护 fan | `locate(fan) → inspect(fan) → clean_blades(fan) → maintain(fan) → confirm_state(fan)` |
| 7 | `maintenance_management` | `electric_curtain` | 检查并养护 electric_curtain | `locate(electric_curtain) → inspect(electric_curtain) → service_track(electric_curtain) → maintain(electric_curtain) → confirm_state(electric_curtain)` |
| 8 | `maintenance_management` | `smart_lock` | 检查并养护 smart_lock | `locate(smart_lock) → inspect(smart_lock) → test_lock(smart_lock) → maintain(smart_lock) → confirm_state(smart_lock)` |
| 9 | `maintenance_management` | `flower_pot` | 检查并养护 flower_pot | `locate(flower_pot) → inspect_plant_and_pests(flower_pot) → remove_pests(flower_pot) → organize(flower_pot) → inspect(flower_pot)` |
| 10 | `maintenance_management` | `watering_can` | 检查并养护 watering_can | `locate(watering_can) → grasp(watering_can) → fill(watering_can) → move(flower_pot) → water(flower_pot,watering_can) → place(watering_can)` |

## 娱乐服务 `entertainment_service`

| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |
|---:|---|---|---|---|
| 1 | `entertainment_service` | `remote_control` | 使用 remote_control 提供娱乐服务 | `locate(remote_control) → grasp(remote_control) → aim(remote_control,television) → press(remote_control) → control(remote_control,television) → confirm_state(television)` |
| 2 | `entertainment_service` | `television` | 使用 television 提供娱乐服务 | `locate(television) → turn_on(television) → play(television) → confirm_state(television)` |
| 3 | `entertainment_service` | `laptop` | 使用 laptop 提供娱乐服务 | `locate(laptop) → turn_on(laptop) → play(laptop) → control(laptop) → confirm_state(laptop)` |
| 4 | `entertainment_service` | `book` | 使用 book 提供娱乐服务 | `locate(book) → grasp(book) → move(entertainment_area) → open(book) → present(book)` |
| 5 | `entertainment_service` | `mobile_phone` | 使用 mobile_phone 提供娱乐服务 | `locate(mobile_phone) → turn_on(mobile_phone) → play(mobile_phone) → control(mobile_phone) → confirm_state(mobile_phone)` |
| 6 | `entertainment_service` | `toy` | 使用 toy 提供娱乐服务 | `locate(toy) → grasp(toy) → move(entertainment_area) → play(toy) → confirm_state(toy)` |
| 7 | `entertainment_service` | `game_controller` | 使用 game_controller 提供娱乐服务 | `locate(game_controller) → grasp(game_controller) → connect(game_controller,entertainment_device) → start_game(entertainment_device) → control(game_controller,entertainment_device) → place(game_controller)` |
| 8 | `entertainment_service` | `speaker` | 使用 speaker 提供娱乐服务 | `locate(speaker) → turn_on(speaker) → connect(speaker,audio_source) → play(speaker) → adjust_volume(speaker)` |

