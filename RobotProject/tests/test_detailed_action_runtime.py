import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.detailed_action_runtime import (
    DetailedActionParser,
    _has_cancelling_negation,
    _has_unresolved_choice,
    _is_information_request,
    _is_completed_statement,
    _is_restrictive_attribute_command,
    format_detailed_action_result,
)
from tools.detailed_action_semantic_data import normalize_semantic_text


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_PATH = ROOT / "tests" / "fixtures" / "detailed_action_runtime_acceptance.json"
TRAIN_PATH = ROOT / "datasets" / "detailed_action_semantic_train.json"
DEV_PATH = ROOT / "datasets" / "detailed_action_semantic_dev.json"
CATALOG_PATH = ROOT / "meta" / "detailed_action_entry_catalog.json"
CHECKPOINT_PATH = ROOT / "models" / "detailed_action_semantic" / "best.pt"


class DetailedActionRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = DetailedActionParser(device="cpu")
        cls.acceptance = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    def test_unresolved_choice_requires_uncertainty_and_alternatives(self):
        self.assertTrue(_has_unresolved_choice("轻柔洗还是羊毛洗，我还没定"))
        self.assertTrue(_has_unresolved_choice("代码没写清是低电量还是校准异常"))
        self.assertFalse(_has_unresolved_choice("把灯调暗一点，别刺眼"))
        self.assertFalse(_has_unresolved_choice("清洗还是整理"))

    def test_unresolved_choice_covers_spoken_missing_detail_variants(self):
        self.assertTrue(_has_unresolved_choice("改成处理杯子，但具体怎么弄我还没决定"))
        self.assertTrue(_has_unresolved_choice("放到方便的位置，可哪个位置我没说清"))
        self.assertTrue(_has_unresolved_choice("擦拭、收纳还是搬走我还没选"))
        self.assertFalse(_has_unresolved_choice("改成擦净杯子，最后放到右侧托盘"))

    def test_restrictive_attribute_commands_are_not_cancellations(self):
        self.assertTrue(_is_restrictive_attribute_command("请把不带电池的设备放到架上"))
        self.assertTrue(_is_restrictive_attribute_command("将非壁挂式屏幕安放在柜面"))
        self.assertTrue(_is_restrictive_attribute_command("把未削的文具收进盒中"))
        self.assertFalse(_is_restrictive_attribute_command("设备不要放在架上，其他位置没说"))
        self.assertFalse(_is_restrictive_attribute_command("屏幕不能朝向窗户"))

    def test_information_request_does_not_match_long_procedural_state_checks(self):
        self.assertTrue(_is_information_request("怎么清洁这块地毯？"))
        self.assertTrue(_is_information_request("空调刚才已经关好了"))
        self.assertFalse(
            _is_information_request(
                "检查摄像头是否已经断电，确认周围没有障碍后再拆下并收好"
            )
        )

    def test_completed_statement_excludes_followup_commands(self):
        self.assertTrue(_is_completed_statement("洗衣机里的床单刚洗完，正在晾着"))
        self.assertTrue(_is_completed_statement("碗筷都洗净并沥干了"))
        self.assertTrue(_is_completed_statement("备用电池都分类收好了"))
        self.assertFalse(_is_completed_statement("把已经发干的杯子仔细擦洗"))
        self.assertFalse(_is_completed_statement("饮料罐已经废弃，先倒净再回收"))

    def test_short_cancel_commands_cover_unlisted_predicates_without_rejecting_constraints(self):
        self.assertTrue(_has_cancelling_negation("洗碗机这次禁止运行"))
        self.assertTrue(_has_cancelling_negation("不要让机器人去给加湿器加水"))
        self.assertTrue(_has_cancelling_negation("这回不要给地毯吸尘"))
        self.assertTrue(_has_cancelling_negation("热水器的加热预约别保留"))
        self.assertFalse(_has_cancelling_negation("擦桌子时别刮伤涂层"))
        self.assertFalse(
            _has_cancelling_negation(
                "处理温度计时先检查外观和电量，异常时停止操作并记录上报"
            )
        )
        self.assertFalse(
            _has_cancelling_negation(
                "这只马克杯的釉面别用硬刷，换温和工具把杯内外擦洗一遍"
            )
        )

    def test_independent_acceptance_set_is_not_train_or_dev_text(self):
        generated = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
        generated += json.loads(DEV_PATH.read_text(encoding="utf-8"))
        generated_texts = {normalize_semantic_text(row["text"]) for row in generated}
        acceptance_texts = {
            normalize_semantic_text(row["text"])
            for group in self.acceptance.values()
            for row in group
        }
        self.assertEqual(len(acceptance_texts), sum(map(len, self.acceptance.values())))
        self.assertTrue(acceptance_texts.isdisjoint(generated_texts))
        runtime_source = (ROOT / "tools" / "detailed_action_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertTrue(
            all(row["text"] not in runtime_source for group in self.acceptance.values() for row in group)
        )

    def test_development_corpus_outputs_are_catalog_bounded_and_rejections_are_safe(self):
        results = [
            (row, self.parser.resolve(row["text"]))
            for row in self.acceptance["decidable"]
        ]
        catalog_labels = {
            row["label"]
            for row in json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["entries"]
        }
        self.assertTrue(
            all(result.get("label") in catalog_labels or result["result_type"] == "clarification_required"
                for _, result in results)
        )

        rejected = [self.parser.resolve(row["text"]) for row in self.acceptance["reject"]]
        self.assertTrue(all(result["result_type"] == "clarification_required" for result in rejected))
        self.assertTrue(all("detailed_actions" not in json.dumps(result, ensure_ascii=False) for result in rejected))

    def test_required_scene_semantics_and_no_action_leaks(self):
        bedtime = self.parser.resolve("我想睡了，把卧室调成睡眠环境")
        self.assertEqual(bedtime["result_type"], "coordination_detailed_action")
        self.assertEqual(
            {item["object_name_zh"] for item in bedtime["objects"]},
            {"空调", "电动窗帘", "卧室灯"},
        )
        floor = self.parser.resolve("地板和地面都该好好清扫了")
        self.assertEqual(floor["result_type"], "coordination_detailed_action")
        self.assertTrue(any("地面" in item["object_name_zh"] or "地板" in item["object_name_zh"] for item in floor["objects"]))
        breakfast = self.parser.resolve("早饭帮我准备妥当")
        self.assertEqual(breakfast["result_type"], "coordination_detailed_action")
        self.assertFalse(any("书" in item["object_name_zh"] for item in breakfast["objects"]))

        serialized = json.dumps([bedtime, floor, breakfast], ensure_ascii=False)
        self.assertNotIn("original_actions", serialized)
        self.assertNotIn("final_actions", serialized)
        self.assertNotIn("expected_actions", serialized)

    def test_explicit_bedtime_catalog_command_returns_only_three_device_details(self):
        result = self.parser.resolve("我要睡觉了")

        self.assertEqual(result["result_type"], "coordination_detailed_action")
        self.assertEqual(
            {item["object_name_zh"] for item in result["objects"]},
            {"空调", "电动窗帘", "卧室灯"},
        )
        self.assertEqual(len(result["objects"]), 3)
        self.assertTrue(all(item["detailed_actions"] for item in result["objects"]))
        serialized = json.dumps(result, ensure_ascii=False)
        for legacy_field in (
            "original_actions",
            "base_actions",
            "final_actions",
            "expected_actions",
            "template_steps",
            "detailed_action_sequence",
        ):
            self.assertNotIn(legacy_field, serialized)

    def test_bedtime_catalog_command_questions_never_execute(self):
        for text in (
            "我要睡觉了？",
            "我要睡觉了?",
            "我要睡觉了吗",
            "我要睡觉了吗？",
            "我要睡觉了么",
        ):
            with self.subTest(text=text):
                result = self.parser.resolve(text)
                self.assertEqual(result["result_type"], "clarification_required")
                self.assertNotIn("objects", result)
                self.assertNotIn("detailed_actions", result)

    def test_ambiguous_low_margin_negation_and_empty_inputs_never_guess(self):
        for text in ("处理一下碗", "处理一下垃圾桶", "别打开窗帘", "", "随便弄一下"):
            with self.subTest(text=text):
                result = self.parser.resolve(text)
                self.assertEqual(result["result_type"], "clarification_required")
                self.assertLessEqual(len(result["candidates"]), 3)
                self.assertNotIn("objects", result)
                self.assertNotIn("detailed_actions", result)

    def test_clause_level_negation_distinguishes_cancellation_from_constraints(self):
        cancellations = (
            "请不要打开空调",
            "空调不要打开",
            "我不想打开空调",
            "麻烦先别开空调",
        )
        for text in cancellations:
            with self.subTest(text=text):
                self.assertEqual(
                    self.parser.resolve(text)["result_type"],
                    "clarification_required",
                )

        for text in (
            "不要紧，帮我打开空调",
            "把锅洗干净，别刮伤涂层",
            "告别客厅前把空调关闭",
            "别墅客厅的空调打开",
            "检查差别后打开空调",
        ):
            with self.subTest(text=text):
                self.assertNotEqual(
                    self.parser.resolve(text)["result_type"],
                    "clarification_required",
                )

    def test_information_questions_and_statements_never_execute_embedded_actions(self):
        for text in (
            "请告诉我如何清洁木地板",
            "空调怎么打开",
            "为何要关闭窗帘",
            "能否介绍一下怎样清洗餐具",
            "我想了解地板清洁的方法",
            "空调今天已经打开过了",
            "木地板刚刚清洁完了",
        ):
            with self.subTest(text=text):
                result = self.parser.resolve(text)
                self.assertEqual(result["result_type"], "clarification_required")
                self.assertNotIn("objects", result)
                self.assertNotIn("detailed_actions", result)

    def test_generic_predicate_complements_and_destinations_are_executable(self):
        cases = {
            "把木地板上的脚印擦掉": "object:original:wood_floor",
            "请把这本书拿到沙发旁边": "object:original:book",
            "把空调调低一点": "object:original:air_conditioner",
        }
        for text, label in cases.items():
            with self.subTest(text=text):
                result = self.parser.resolve(text)
                self.assertEqual(result.get("label"), label, result)

    def test_single_explicit_catalog_object_cannot_be_stolen_by_scene_routing(self):
        cases = {
            "把浴室地面擦干净": "object:original:bathroom_floor",
            "请清理地毯表面的灰尘": "object:original:carpet",
            "把淋浴器刷洗一下": "object:original:shower",
            "把肥皂收进洗漱用品区": "object:original:soap",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = self.parser.resolve(text)
                self.assertEqual(result.get("label"), expected, result)

    def test_task2_dev_aliases_are_catalog_label_surfaces_not_freehand_runtime_terms(self):
        cases = {
            "把层压板面擦拭干净": "object:original:laminate_floor",
            "让自动吸尘机清扫地面": "object:original:robot_vacuum",
            "把带把杯洗净": "object:original:mug",
            "将废物筒清空": "object:original:trash_bin",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.parser.resolve(text).get("label"), expected)

    def test_duplicate_catalog_names_are_disambiguated_by_operation_and_purpose(self):
        cases = {
            "把碗上的油污洗掉": "object:original:bowl",
            "拿碗盛好刚炒熟的菜": "object:vla:vla_017",
            "把锅里的热汤盛到碗中": "object:vla:vla_017",
            "把锅刷干净": "object:original:pot",
            "用锅把汤煮开": "object:vla:vla_021",
            "清洗马克杯": "object:original:mug",
            "把马克杯递到我手边": "object:vla:vla_028",
            "把盘子洗净": "object:original:plate",
            "把盘子摆到餐桌座位前": "object:vla:vla_058",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = self.parser.resolve(text)
                self.assertEqual(result.get("label"), expected, result)

        for text in ("处理碗", "处理锅", "处理马克杯", "处理盘子"):
            with self.subTest(text=text):
                self.assertEqual(self.parser.resolve(text)["result_type"], "clarification_required")

    def test_operation_task_matrix_rejects_incompatible_object_actions(self):
        for text in (
            "把书放进烤箱烤熟",
            "用灭火器煮一锅汤",
            "拿空调当餐具摆到桌上",
            "用枕头测量体温",
        ):
            with self.subTest(text=text):
                result = self.parser.resolve(text)
                self.assertEqual(result["result_type"], "clarification_required")
                self.assertNotIn("objects", result)
                self.assertNotIn("detailed_actions", result)

    def test_negation_uses_lexical_boundaries_and_preserves_restrictive_clauses(self):
        for text in (
            "把餐具分别清洗干净",
            "把特别脏的地板清洁一下",
            "识别空调后把它关闭",
            "按类别整理衣物",
        ):
            with self.subTest(text=text):
                result = self.parser.resolve(text)
                self.assertNotIn("否定", result.get("reason", ""))

        for text in (
            "把锅洗净，清洗时不要移动旁边的杯子",
            "清洗锅时不要刮伤涂层",
        ):
            with self.subTest(text=text):
                self.assertNotEqual(self.parser.resolve(text)["result_type"], "clarification_required")

    def test_completed_aspect_variants_are_not_new_commands(self):
        for text in (
            "空调刚刚关掉了",
            "地板已经擦完啦",
            "窗帘现在开着呢",
            "我刚才洗过这个碗",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.parser.resolve(text)["result_type"], "clarification_required")

    def test_runtime_language_resources_have_declared_non_acceptance_sources(self):
        source = (ROOT / "tools" / "detailed_action_runtime.py").read_text(encoding="utf-8")
        for forbidden_name in (
            "RUNTIME_REPLACEMENTS",
            "NON_EXECUTABLE_TERMS",
            "SCENE_SCOPE_TERMS",
            "TASK_TERMS",
            "SCENE_CONCEPT_FAMILIES",
        ):
            self.assertNotIn(forbidden_name, source)
        self.assertIn("Source: Task 2 train/dev", source)
        self.assertIn("Source: general Mandarin grammar", source)
        contaminated_fragments = (
            "晨间餐点", "中午这顿饭", "晚上的饭菜", "刷洗消毒", "安防自检",
            "居家健康检测", "入睡条件", "键鼠", "洗衣设备", "消防器材",
            "厨房里夹子", "住处安全", "准备办公", "开始学习", "给客人",
            "马上出门", "冷水轻柔", "避免高温损伤", "分类后重新收纳",
        )
        self.assertTrue(all(fragment not in source for fragment in contaminated_fragments))
        self.assertNotIn("detailed_action_runtime_blind_v2", source)

    def test_configurable_confidence_and_margin_thresholds_fail_closed(self):
        strict_confidence = DetailedActionParser(device="cpu", minimum_confidence=0.99)
        low_confidence = strict_confidence.resolve("空调调凉快一点")
        self.assertEqual(low_confidence["result_type"], "clarification_required")
        self.assertIn("置信度", low_confidence["reason"])

        strict_margin = DetailedActionParser(device="cpu", minimum_margin=0.99)
        near_candidates = strict_margin.resolve("准备睡觉")
        self.assertEqual(near_candidates["result_type"], "clarification_required")
        self.assertIn("候选过近", near_candidates["reason"])

    def test_checkpoint_catalog_hash_and_label_audit_fail_closed(self):
        payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        for field, replacement in (
            ("catalog_sha256", "0" * 64),
            ("labels", payload["labels"][:-1]),
        ):
            broken = copy.deepcopy(payload)
            broken[field] = replacement
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "broken.pt"
                torch.save(broken, checkpoint)
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "checkpoint.*catalog"):
                        DetailedActionParser(checkpoint_path=checkpoint, device="cpu")

        reordered = copy.deepcopy(payload)
        reordered["labels"][0], reordered["labels"][1] = (
            reordered["labels"][1], reordered["labels"][0]
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "reordered.pt"
            torch.save(reordered, checkpoint)
            with self.assertRaisesRegex(ValueError, "checkpoint.*catalog"):
                DetailedActionParser(checkpoint_path=checkpoint, device="cpu")

    def test_real_model_changes_candidate_internal_order(self):
        labels = ("object:original:bowl", "object:vla:vla_017")
        with_model = self.parser.rank_candidate_labels("用碗装苹果", labels)
        without_model = self.parser.rank_candidate_labels(
            "用碗装苹果", labels, model_weight=0.0
        )
        self.assertNotEqual(
            [row["label"] for row in with_model],
            [row["label"] for row in without_model],
        )
        self.assertTrue(all(row["model_score"] >= 0 for row in with_model))

    def test_real_model_changes_end_to_end_resolve_confidence_on_independent_dev_text(self):
        without_model = DetailedActionParser(device="cpu", model_weight=0.0)
        text = "请开启客厅里的空调"
        normal = self.parser.resolve(text)
        disabled = without_model.resolve(text)
        self.assertEqual(normal.get("label"), "object:original:air_conditioner")
        self.assertEqual(disabled.get("label"), normal.get("label"))
        self.assertNotEqual(normal["confidence"], disabled["confidence"])

    def test_default_thresholds_and_equality_boundaries(self):
        natural_rejections = (
            "把厨房弄一下",
            "清洁一下",
            "安排一下",
        )
        self.assertTrue(
            all(
                self.parser.resolve(text)["result_type"] == "clarification_required"
                for text in natural_rejections
            )
        )

        baseline = self.parser.resolve("空调调凉快一点")
        confidence = baseline["confidence"]
        at_threshold = DetailedActionParser(device="cpu", minimum_confidence=confidence)
        above_threshold = DetailedActionParser(
            device="cpu", minimum_confidence=confidence + 0.0002
        )
        self.assertEqual(
            at_threshold.resolve("空调调凉快一点")["result_type"],
            "object_detailed_action",
        )
        self.assertEqual(
            above_threshold.resolve("空调调凉快一点")["result_type"],
            "clarification_required",
        )

    def test_formatter_separates_success_from_clarification(self):
        success = self.parser.resolve("空调调凉快一点")
        rendered = format_detailed_action_result(success)
        self.assertIn("输入", rendered)
        self.assertIn("识别类别", rendered)
        self.assertIn("额外细分动作", rendered)
        self.assertIn("必要参数", rendered)

        clarification = self.parser.resolve("处理一下碗")
        rendered_clarification = format_detailed_action_result(clarification)
        self.assertIn("澄清原因", rendered_clarification)
        self.assertIn("候选", rendered_clarification)
        self.assertNotIn("细分动作", rendered_clarification)


if __name__ == "__main__":
    unittest.main()
