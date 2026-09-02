import unittest

from tools.multi_device_coordination import (
    format_multi_device_result,
    is_multi_device_instruction,
    plan_multi_device_instruction,
)


BEDTIME_INSTRUCTION = (
    "准备就寝环境：空调设为24℃、关闭电动窗帘、"
    "将卧室灯调至低亮度，并检查各设备状态。"
)


class MultiDeviceCoordinationTest(unittest.TestCase):
    def test_recognizes_supported_bedtime_instruction_only(self):
        self.assertTrue(is_multi_device_instruction(BEDTIME_INSTRUCTION))
        self.assertTrue(is_multi_device_instruction("请准备就寝环境"))
        self.assertFalse(is_multi_device_instruction("把水杯拿给我"))

    def test_bedtime_plan_coordinates_existing_devices_without_conflicts(self):
        result = plan_multi_device_instruction(
            BEDTIME_INSTRUCTION,
            environment={"indoor_temperature_c": 25.0},
        )

        self.assertEqual(result["result_type"], "multi_device_coordination")
        self.assertEqual(result["scenario"], "bedtime_environment")
        self.assertEqual(
            set(result["devices"]),
            {"air_conditioner", "electric_curtain", "bedroom_lamp"},
        )
        self.assertIn(
            "set_temperature(air_conditioner,24C)", result["final_actions"]
        )
        self.assertIn("close(electric_curtain)", result["final_actions"])
        self.assertIn(
            "set_brightness(bedroom_lamp,20%)", result["final_actions"]
        )
        self.assertTrue(result["conflict_check"]["passed"])
        self.assertTrue(result["completed"])
        self.assertFalse(result["replanning"]["triggered"])
        self.assertEqual(
            [
                (
                    subtask["resolved"]["task"],
                    subtask["resolved"]["object"],
                    subtask["resolved"]["operation"],
                )
                for subtask in result["subtasks"]
            ],
            [
                ("appliance_management", "air_conditioner", "turn_on"),
                ("appliance_management", "electric_curtain", "close"),
                ("bedroom_service", "bedroom_lamp", "turn_on"),
            ],
        )
        self.assertTrue(
            all(
                subtask["planner"] == "existing_manual_instruction_pipeline"
                for subtask in result["subtasks"]
            )
        )

    def test_electric_curtain_close_uses_close_action_not_power_off(self):
        result = plan_multi_device_instruction(BEDTIME_INSTRUCTION)
        curtain = next(
            subtask
            for subtask in result["subtasks"]
            if subtask["resolved"]["object"] == "electric_curtain"
        )

        self.assertIn("close(electric_curtain)", curtain["base_actions"])
        self.assertNotIn("turn_off(electric_curtain)", curtain["base_actions"])

    def test_temperature_change_triggers_cooling_replan(self):
        result = plan_multi_device_instruction(
            BEDTIME_INSTRUCTION,
            environment={
                "indoor_temperature_c": 31.0,
                "change_reason": "室温临时升高",
            },
        )

        self.assertTrue(result["replanning"]["triggered"])
        self.assertEqual(result["replanning"]["air_conditioner_mode"], "cool")
        self.assertIn("set_mode(air_conditioner,cool)", result["final_actions"])

    def test_temperature_change_triggers_heating_replan(self):
        result = plan_multi_device_instruction(
            BEDTIME_INSTRUCTION,
            environment={
                "indoor_temperature_c": 17.0,
                "change_reason": "夜间降温",
            },
        )

        self.assertTrue(result["replanning"]["triggered"])
        self.assertEqual(result["replanning"]["air_conditioner_mode"], "heat")
        self.assertIn("set_mode(air_conditioner,heat)", result["final_actions"])

    def test_formatter_reports_real_device_targets_and_replanning(self):
        result = plan_multi_device_instruction(
            BEDTIME_INSTRUCTION,
            environment={
                "indoor_temperature_c": 31.0,
                "change_reason": "室温临时升高",
            },
        )

        text = format_multi_device_result(result)

        self.assertIn("空调目标温度：24℃", text)
        self.assertIn("电动窗帘：关闭", text)
        self.assertIn("卧室灯亮度：20%", text)
        self.assertIn("重新规划：是", text)
        self.assertIn("冲突检查：通过", text)
        self.assertIn("子任务1：appliance_management / air_conditioner", text)
        self.assertIn("子任务2：appliance_management / electric_curtain", text)
        self.assertIn("子任务3：bedroom_service / bedroom_lamp", text)


if __name__ == "__main__":
    unittest.main()
