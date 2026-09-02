import unittest
from pathlib import Path

from tools.detailed_action_runtime import DetailedActionParser
from tools.detailed_action_runtime import (
    _has_object_attribute_mismatch,
    _has_unresolved_instance_reference,
    _has_unresolved_choice,
    _is_out_of_domain_request,
    _semantic_text,
)


class DetailedActionRuntimeBgeTest(unittest.TestCase):
    def test_extended_unresolved_choice_and_out_of_domain_guards(self):
        self.assertTrue(_has_unresolved_choice("今晚该制冷还是除湿，我没拿准"))
        self.assertTrue(_has_unresolved_choice("还是拉上吧，我想不清最后要哪种状态"))
        self.assertTrue(_has_unresolved_choice("究竟哪一本再说"))
        self.assertTrue(_is_out_of_domain_request("将商标申请材料递交到知识产权在线平台"))
        self.assertFalse(_is_out_of_domain_request("把书放回书架"))

    def test_unresolved_instance_reference_requires_location(self):
        self.assertTrue(_has_unresolved_instance_reference("钥匙该收起来了"))
        self.assertTrue(_has_unresolved_instance_reference("拖鞋需要整理"))
        self.assertFalse(_has_unresolved_instance_reference("把茶几上的钥匙拿来"))
        self.assertFalse(_has_unresolved_instance_reference("把钥匙、手机和钱包集中备好"))
        self.assertFalse(_has_unresolved_instance_reference("把两对凉拖鞋清净后分格齐放"))

    def test_object_attribute_mismatch_has_safe_positive_controls(self):
        self.assertTrue(_has_object_attribute_mismatch("把披萨刀的摄像头对焦到桌面"))
        self.assertTrue(_has_object_attribute_mismatch("枕头的风速调到三档"))
        self.assertTrue(_has_object_attribute_mismatch("吸顶灯的打印纸尺寸改成A4"))
        self.assertTrue(_has_object_attribute_mismatch("让书架连接家里的 Wi-Fi 并开始上网"))
        self.assertTrue(_has_object_attribute_mismatch("给牙刷设定一条扫地路线"))
        self.assertFalse(_has_object_attribute_mismatch("把数码相机的摄像头对焦到桌面"))
        self.assertFalse(_has_object_attribute_mismatch("把风扇的风速调到三档"))
        self.assertFalse(_has_object_attribute_mismatch("打印机的打印纸尺寸改成A4"))
        self.assertFalse(_has_object_attribute_mismatch("把客厅音箱的音量调小"))

    def test_bge_backend_loads_locally_and_resolves(self):
        parser = DetailedActionParser(semantic_backend="bge", device="cpu")
        result = parser.resolve("请把牙刷立放进漱口杯")
        self.assertEqual(result["result_type"], "object_detailed_action")
        self.assertEqual(result["label"], "object:vla:vla_080")
        self.assertTrue(result["model_used"])
        self.assertIsNotNone(parser.bge_object_operation_specialist)
        self.assertEqual(
            parser.bge_payload["object_operation_specialist"]["format"],
            "object_operation_specialist_v2_ontology_augmented",
        )
        self.assertEqual(
            parser.bge_payload["short_object_gate"],
            {"maximum_chars": 12, "additional_margin": 0.5},
        )

    def test_exact_reviewed_scene_command_bypasses_learned_gate(self):
        parser = DetailedActionParser(semantic_backend="bge", device="cpu")
        result = parser.resolve("我要睡觉了")
        self.assertEqual(result["result_type"], "coordination_detailed_action")
        self.assertEqual(result["label"], "scene:MDC-033")

    def test_calibrated_scene_paraphrase_reaches_coordination_result(self):
        parser = DetailedActionParser(semantic_backend="bge", device="cpu")
        result = parser.resolve(
            "厨房大扫除要开工了，先把海绵送到操作台，其他清洁工具也配好"
        )
        self.assertEqual(result["result_type"], "coordination_detailed_action")
        self.assertEqual(result["label"], "scene:MDC-008")

    def test_calibrated_object_detail_stays_single_object(self):
        parser = DetailedActionParser(semantic_backend="bge", device="cpu")
        result = parser.resolve("将备用床垫归位到储物间预留的床垫架上")
        self.assertEqual(result["result_type"], "object_detailed_action")
        self.assertEqual(result["label"], "object:original:mattress")

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            DetailedActionParser(semantic_backend="unknown")

    def test_bge_backend_loads_embedded_character_gate(self):
        root = Path(__file__).resolve().parents[1]
        parser = DetailedActionParser(
            semantic_backend="bge",
            bge_checkpoint_path=root / "models/detailed_action_bge/best_cls_anchor5_rerank_character_gate.pt",
            device="cpu",
        )
        self.assertEqual(parser.bge_gate_backend, "character_ngram")

    def test_bge_backend_loads_embedded_object_operation_specialist(self):
        root = Path(__file__).resolve().parents[1]
        parser = DetailedActionParser(
            semantic_backend="bge",
            bge_checkpoint_path=root / "models/detailed_action_bge/best_hybrid_dual_intent_object_operation.pt",
            device="cpu",
        )
        self.assertIsNotNone(parser.bge_object_operation_specialist)
        self.assertEqual(len(parser.bge_object_operation_capability_labels), 205)

    def test_object_operation_specialist_rejects_incompatible_request(self):
        root = Path(__file__).resolve().parents[1]
        parser = DetailedActionParser(
            semantic_backend="bge",
            bge_checkpoint_path=root / "models/detailed_action_bge/best_hybrid_dual_intent_object_operation.pt",
            device="cpu",
        )
        instruction = "把披萨刀的摄像头对焦到桌面上"
        matched = parser._matched_objects(_semantic_text(instruction))
        self.assertFalse(parser._object_operation_specialist_allows(instruction, matched))

    def test_same_surface_objects_require_disambiguating_context(self):
        parser = DetailedActionParser(semantic_backend="bge", device="cpu")
        result = parser.resolve("钥匙该收起来了，帮我收好。")
        self.assertEqual(result["result_type"], "clarification_required")
        self.assertIn("多个同名对象", result["reason"])


if __name__ == "__main__":
    unittest.main()
