import unittest

from tools.manual_instruction_entry import (
    NonExecutableInstructionError,
    build_manual_result,
    describe_actions,
    describe_resolved_intent,
    find_expected_actions,
    format_result_text,
    load_benchmark_records,
    resolve_manual_instruction,
)


class ManualInstructionEntryTest(unittest.TestCase):
    def test_display_helpers_translate_reading_lamp_actions(self):
        resolved = {
            "task": "appliance_management",
            "object": "reading_lamp",
            "operation": "turn_off",
        }

        self.assertEqual(describe_resolved_intent(resolved), "关闭阅读灯")
        self.assertEqual(
            describe_actions(
                [
                    "locate(reading_lamp)",
                    "turn_off(reading_lamp)",
                    "confirm_state(reading_lamp)",
                ]
            ),
            ["找到阅读灯", "关闭阅读灯", "确认已经关闭"],
        )

    def test_generic_light_power_commands_use_ceiling_light(self):
        cases = (
            ("开灯", "turn_on", ["locate(ceiling_light)", "turn_on(ceiling_light)", "inspect(ceiling_light)"]),
            ("关灯", "turn_off", ["locate(ceiling_light)", "turn_off(ceiling_light)", "inspect(ceiling_light)"]),
        )
        records = load_benchmark_records()
        for text, operation, actions in cases:
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, "appliance_management")
                self.assertEqual(resolved.object, "ceiling_light")
                self.assertEqual(resolved.operation, operation)
                self.assertEqual(find_expected_actions(resolved, records), actions)

    def test_television_power_commands_have_entertainment_action_plans(self):
        cases = (
            ("打开电视", "turn_on", ["locate(television)", "turn_on(television)", "confirm_state(television)"]),
            ("关闭电视", "turn_off", ["locate(television)", "turn_off(television)", "confirm_state(television)"]),
        )
        records = load_benchmark_records()
        for text, operation, actions in cases:
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, "entertainment_service")
                self.assertEqual(resolved.object, "television")
                self.assertEqual(resolved.operation, operation)
                self.assertEqual(find_expected_actions(resolved, records), actions)

    def test_common_short_power_commands_resolve_without_ambiguity(self):
        cases = (
            ("开台灯", "appliance_management", "reading_lamp", "turn_on"),
            ("关台灯", "appliance_management", "reading_lamp", "turn_off"),
            ("开风扇", "appliance_management", "fan", "turn_on"),
            ("关风扇", "appliance_management", "fan", "turn_off"),
            ("开洗衣机", "appliance_management", "washing_machine", "turn_on"),
            ("关洗衣机", "appliance_management", "washing_machine", "turn_off"),
            ("开卧室灯", "bedroom_service", "bedroom_lamp", "turn_on"),
            ("关卧室灯", "bedroom_service", "bedroom_lamp", "turn_off"),
        )
        records = load_benchmark_records()
        for text, task, obj, operation in cases:
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual((resolved.task, resolved.object, resolved.operation), (task, obj, operation))
                actions = find_expected_actions(resolved, records)
                self.assertIn(f"{operation}({obj})", actions)
                opposite = "turn_off" if operation == "turn_on" else "turn_on"
                self.assertNotIn(f"{opposite}({obj})", actions)

    def test_window_commands_preserve_distinct_operations_and_actions(self):
        cases = (
            ("开窗户", "open", ["locate(window)", "open(window)", "inspect(window)"]),
            ("关窗户", "close", ["locate(window)", "close(window)", "inspect(window)"]),
            ("擦窗户", "clean", ["locate(window)", "wipe(window)", "inspect(window)"]),
        )
        records = load_benchmark_records()
        for text, operation, actions in cases:
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, "window_care")
                self.assertEqual(resolved.object, "window")
                self.assertEqual(resolved.operation, operation)
                self.assertEqual(find_expected_actions(resolved, records), actions)

    def test_air_conditioner_commands_preserve_operation_and_final_state(self):
        cases = (
            ("开空调", "turn_on", ["locate(air_conditioner)", "turn_on(air_conditioner)", "inspect(air_conditioner)"]),
            ("关空调", "turn_off", ["locate(air_conditioner)", "turn_off(air_conditioner)", "inspect(air_conditioner)"]),
            ("检查空调滤网", "maintain", ["locate(air_conditioner)", "inspect(air_conditioner)", "clean_filter(air_conditioner)", "maintain(air_conditioner)", "confirm_state(air_conditioner)"]),
        )
        records = load_benchmark_records()
        for text, operation, actions in cases:
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.operation, operation)
                self.assertEqual(find_expected_actions(resolved, records), actions)

    def test_pour_water_variants_include_fill_and_pour_actions(self):
        expected = [
            "locate(water_cup)",
            "grasp(water_cup)",
            "move(water_source)",
            "fill(water_cup)",
            "pour(water_cup)",
            "move(person)",
            "place(water_cup)",
        ]
        records = load_benchmark_records()
        for text in ("帮我倒杯水", "倒杯水", "倒一杯水", "给我倒水", "接杯水"):
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, "food_serving")
                self.assertEqual(resolved.object, "water_cup")
                self.assertEqual(resolved.operation, "pour_water")
                self.assertEqual(find_expected_actions(resolved, records), expected)

    def test_pour_water_result_includes_chinese_action_parse(self):
        resolved = resolve_manual_instruction("倒杯水")
        actions = find_expected_actions(resolved, load_benchmark_records())

        result = build_manual_result("倒杯水", resolved, actions, actions)

        self.assertIn("接水/注水", result["action_parse"]["action_primitives"])
        self.assertIn("倒水", result["action_parse"]["action_primitives"])

    def test_routes_air_conditioner_power_commands_to_appliance_management(self):
        for text in ("开空调", "打开空调", "把空调打开", "启动空调", "关空调", "关闭空调"):
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, "appliance_management")
                self.assertEqual(resolved.object, "air_conditioner")

    def test_routes_air_conditioner_maintenance_commands_to_maintenance(self):
        for text in ("维护空调", "检查空调滤网", "保养空调"):
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, "maintenance_management")
                self.assertEqual(resolved.object, "air_conditioner")

    def test_routes_implicit_water_pouring_commands_to_water_serving(self):
        for text in ("帮我倒杯水", "倒一杯水", "接杯水", "给我倒水"):
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, "food_serving")
                self.assertEqual(resolved.object, "water_cup")

    def test_routes_water_cup_commands_by_verb_intent(self):
        cases = (
            ("洗水杯", "dish_washing", "cup"),
            ("把杯子洗一下", "dish_washing", "cup"),
            ("把水杯拿给我", "object_fetching", "water_cup"),
            ("给我送杯水", "food_serving", "water_cup"),
            ("递杯水给我", "food_serving", "water_cup"),
        )
        for text, task, obj in cases:
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, task)
                self.assertEqual(resolved.object, obj)

    def test_cross_intent_water_cup_fetch_has_expected_actions(self):
        resolved = resolve_manual_instruction("把水杯拿给我")

        self.assertEqual(
            find_expected_actions(resolved, load_benchmark_records()),
            [
                "locate(water_cup)",
                "grasp(water_cup)",
                "move(person)",
                "release(water_cup)",
            ],
        )

    def test_air_conditioner_maintenance_has_expected_actions(self):
        resolved = resolve_manual_instruction("检查空调滤网")

        self.assertEqual(
            find_expected_actions(resolved, load_benchmark_records()),
            [
                "locate(air_conditioner)",
                "inspect(air_conditioner)",
                "clean_filter(air_conditioner)",
                "maintain(air_conditioner)",
                "confirm_state(air_conditioner)",
            ],
        )

    def test_routes_shared_objects_by_action_intent(self):
        examples = [
            ("把遥控器拿给我", "object_fetching", "remote_control"),
            ("用遥控器打开电视", "entertainment_service", "remote_control"),
            ("打开电视播放节目", "entertainment_service", "television"),
        ]
        for text, task, obj in examples:
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, task)
                if obj:
                    self.assertEqual(resolved.object, obj)

    def test_routes_entertainment_power_vocabulary(self):
        for text in ("关闭电视", "电视开机", "电视关机", "打开电视电源"):
            with self.subTest(text=text):
                self.assertEqual(resolve_manual_instruction(text).task, "entertainment_service")

    def test_unified_cleaning_wording_keeps_compatible_subtask(self):
        self.assertEqual(resolve_manual_instruction("执行清洁任务，清洁木地板").task, "floor_cleaning")
        self.assertEqual(resolve_manual_instruction("执行清洁任务，把杯子洗一下").task, "dish_washing")

    def test_shared_object_without_intent_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "object_fetching.*entertainment_service"):
            resolve_manual_instruction("遥控器")

    def test_general_top_candidate_tie_lists_distinct_tasks(self):
        with self.assertRaisesRegex(ValueError, "object_fetching.*entertainment_service"):
            resolve_manual_instruction("手机和书")

    def test_cleaning_compatibility_alias_is_not_a_false_ambiguity(self):
        resolved = resolve_manual_instruction("木地板")
        self.assertEqual(resolved.task, "floor_cleaning")

    def test_flower_pot_is_now_a_canonical_executable_object(self):
        for text in ("给盆栽浇水", "帮盆栽浇点水", "给花盆里的植物浇水"):
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, "maintenance_management")
                self.assertEqual(resolved.object, "flower_pot")
                self.assertTrue(resolved.is_executable)
                actions = find_expected_actions(resolved, [])
                self.assertEqual(resolved.operation, "water_plants")
                self.assertIn("water(flower_pot,watering_can)", actions)

    def test_short_flower_care_and_watering_phrases_use_specific_plans(self):
        care = resolve_manual_instruction("照顾花")
        water = resolve_manual_instruction("帮我浇花")

        self.assertEqual(
            (care.task, care.object, care.operation),
            ("maintenance_management", "flower_pot", "care_plant"),
        )
        self.assertIn("remove_pests(flower_pot)", find_expected_actions(care, []))
        self.assertEqual(
            (water.task, water.object, water.operation),
            ("maintenance_management", "flower_pot", "water_plants"),
        )
        self.assertIn("water(flower_pot,watering_can)", find_expected_actions(water, []))

    def test_speaker_and_game_controller_use_object_specific_operations(self):
        speaker = resolve_manual_instruction("打开音响")
        controller = resolve_manual_instruction("处理游戏手柄")

        self.assertEqual(
            (speaker.task, speaker.object, speaker.operation),
            ("entertainment_service", "speaker", "play_audio"),
        )
        self.assertEqual(
            (controller.task, controller.object, controller.operation),
            ("entertainment_service", "game_controller", "operate"),
        )

    def test_popup_text_uses_chinese_primitives_for_new_runtime_plans(self):
        resolved = resolve_manual_instruction("处理游戏手柄")
        actions = find_expected_actions(resolved, [])
        text = format_result_text(
            build_manual_result("处理游戏手柄", resolved, actions, actions)
        )

        self.assertIn("匹配操作：操控游戏手柄", text)
        self.assertIn("定位 → 握取 → 连接设备 → 启动游戏 → 操控游戏 → 轻放", text)
        self.assertNotIn("→ connect →", text)

    def test_new_tasks_never_default_to_unrelated_objects(self):
        for text in ("执行维护管理", "播放节目"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "requires a specific canonical object"):
                    resolve_manual_instruction(text)
    def test_resolves_user_phrase_clean_laundry_to_laundry_task(self):
        resolved = resolve_manual_instruction("清洗衣物")

        self.assertEqual(resolved.task, "laundry")
        self.assertEqual(resolved.object, "child_clothing")
        self.assertEqual(resolved.target, "washer")
        self.assertEqual(resolved.normalized_instruction, "清洗衣物")
        self.assertIn("关键词", resolved.note)

    def test_ironing_bed_sheet_preserves_ironing_semantics(self):
        resolved = resolve_manual_instruction("熨烫床单")

        self.assertEqual(
            (resolved.task, resolved.object, resolved.operation),
            ("clothing_care", "bed_sheet", "iron"),
        )
        self.assertIn("iron(bed_sheet)", find_expected_actions(resolved, []))
        self.assertNotIn("fold(bed_sheet)", find_expected_actions(resolved, []))

    def test_inspecting_blood_pressure_monitor_is_not_a_delivery_plan(self):
        resolved = resolve_manual_instruction("检查血压计")

        self.assertEqual(
            (resolved.task, resolved.object, resolved.operation),
            ("elderly_assistance", "blood_pressure_monitor", "inspect"),
        )
        self.assertEqual(
            find_expected_actions(resolved, []),
            ["locate(blood_pressure_monitor)", "inspect(blood_pressure_monitor)"],
        )

    def test_explicit_table_destination_overrides_food_serving_default(self):
        resolved = resolve_manual_instruction("把餐盘放到桌上")

        self.assertEqual(
            (resolved.task, resolved.object, resolved.target, resolved.operation),
            ("food_serving", "food_tray", "table", "serve"),
        )
        self.assertEqual(
            find_expected_actions(resolved, []),
            ["locate(food_tray)", "grasp(food_tray)", "move(table)", "place(food_tray)"],
        )
        self.assertEqual(
            find_expected_actions(resolved, load_benchmark_records()),
            ["locate(food_tray)", "grasp(food_tray)", "move(table)", "place(food_tray)"],
        )

    def test_dry_is_rendered_as_drying_not_inspection(self):
        resolved = resolve_manual_instruction("清洗衣物")
        actions = find_expected_actions(resolved, [])
        text = format_result_text(build_manual_result("清洗衣物", resolved, actions, actions))

        self.assertIn("晾干", text)
        self.assertNotIn("晾干 → 检查", text)

    def test_washing_cup_keeps_sink_washing_plan_after_task_normalization(self):
        resolved = resolve_manual_instruction("清洗杯子")

        self.assertEqual(
            (resolved.task, resolved.object, resolved.target, resolved.operation),
            ("cleaning", "cup", "sink", "wash"),
        )
        self.assertEqual(
            find_expected_actions(resolved, []),
            [
                "locate(cup)",
                "grasp(cup)",
                "move(sink)",
                "wash(cup)",
                "rinse(cup)",
                "place(cup)",
            ],
        )

    def test_inspecting_iron_is_not_mistaken_for_ironing(self):
        resolved = resolve_manual_instruction("检查熨斗")

        self.assertEqual(
            (resolved.task, resolved.object, resolved.operation),
            ("clothing_care", "iron", "inspect"),
        )
        self.assertEqual(find_expected_actions(resolved, []), ["locate(iron)", "inspect(iron)"])

    def test_resolves_120_scope_phrases_by_intent_and_object(self):
        examples = [
            ("帮我洗鞋", "laundry", "shoes"),
            ("把杯子洗了", "dish_washing", "cup"),
            ("整理一下书架", "organizing", "bookshelf"),
            ("打开阅读灯", "appliance_management", "reading_lamp"),
            ("检查门锁安全", "security_monitoring", "smart_lock"),
            ("把手机拿过来", "object_fetching", "mobile_phone"),
        ]

        for text, task, obj in examples:
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(resolved.task, task)
                self.assertEqual(resolved.object, obj)

    def test_build_manual_result_reports_actions_and_completion(self):
        resolved = resolve_manual_instruction("清洗衣物")
        expected = [
            "locate(child_clothing)",
            "grasp(child_clothing)",
            "move(washer)",
            "wash(child_clothing)",
            "dry(child_clothing)",
        ]

        result = build_manual_result(
            "清洗衣物", resolved, expected, expected, model_used=True
        )

        self.assertTrue(result["completed"])
        self.assertEqual(result["completion_status"], "通过")
        self.assertEqual(result["predicted_actions"], expected)

    def test_format_result_text_contains_actions_for_popup(self):
        resolved = resolve_manual_instruction("把杯子洗了")
        actions = [
            "locate(cup)",
            "grasp(cup)",
            "move(sink)",
            "wash(cup)",
            "rinse(cup)",
            "place(cup)",
        ]
        result = build_manual_result(
            "把杯子洗了", resolved, actions, actions, model_used=True
        )

        text = format_result_text(result)

        self.assertIn("你的指令：把杯子洗了", text)
        self.assertIn("我理解为：清洗杯子。", text)
        self.assertIn("匹配物体：cup", text)
        self.assertIn("匹配操作：清洗 (wash)", text)
        self.assertIn("1. locate(cup)", text)
        self.assertIn("完成情况：通过", text)

    def test_popup_text_keeps_user_words_and_separates_technical_details(self):
        resolved = resolve_manual_instruction("睡前把阅读灯关了")
        actions = [
            "locate(reading_lamp)",
            "turn_off(reading_lamp)",
            "confirm_state(reading_lamp)",
        ]
        result = build_manual_result(
            "睡前把阅读灯关了", resolved, actions, actions, model_used=True
        )

        text = format_result_text(result)
        visible, technical = text.split("技术详情（供检查）：", maxsplit=1)

        self.assertIn("你的指令：睡前把阅读灯关了", visible)
        self.assertIn("我理解为：关闭阅读灯。", visible)
        self.assertIn("建议步骤：找到阅读灯 → 关闭阅读灯 → 确认已经关闭", visible)
        self.assertNotIn("reading_lamp", visible)
        self.assertIn("reading_lamp", technical)

    def test_model_error_cannot_be_reported_as_rule_fallback_pass(self):
        resolved = resolve_manual_instruction("照顾花")
        result = build_manual_result(
            "照顾花",
            resolved,
            ["locate(flower_pot)"],
            ["locate(flower_pot)"],
            model_used=False,
            prediction_error="stale checkpoint",
        )

        self.assertFalse(result["completed"])
        self.assertNotIn("规则回退", result["completion_status"])

    def test_free_form_water_request_resolves_to_pour_water(self):
        for text in ("给我倒杯水", "帮我倒水", "倒点水过来", "我想喝水"):
            with self.subTest(text=text):
                resolved = resolve_manual_instruction(text)
                self.assertEqual(
                    (resolved.task, resolved.object, resolved.operation),
                    ("food_serving", "water_cup", "pour_water"),
                )

    def test_opposite_reading_lamp_instructions_keep_opposite_operations(self):
        off = resolve_manual_instruction("把阅读灯关掉")
        on = resolve_manual_instruction("把阅读灯打开")
        self.assertEqual(
            (off.task, off.object, off.operation),
            ("appliance_management", "reading_lamp", "turn_off"),
        )
        self.assertEqual(
            (on.task, on.object, on.operation),
            ("appliance_management", "reading_lamp", "turn_on"),
        )


if __name__ == "__main__":
    unittest.main()
