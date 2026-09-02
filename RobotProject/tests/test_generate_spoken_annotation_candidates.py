from __future__ import annotations

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock
import warnings

from tools import generate_spoken_annotation_candidates as gen


class CandidateGeneratorTests(unittest.TestCase):
    def test_single_draft_seed_is_reproducibly_derived_from_label_and_index(self) -> None:
        label = "object:original:window"

        first = gen.derive_draft_seed(314159, label, 0)

        self.assertEqual(first, gen.derive_draft_seed(314159, label, 0))
        self.assertNotEqual(first, gen.derive_draft_seed(314159, label, 1))
        self.assertNotEqual(
            first,
            gen.derive_draft_seed(314159, "object:original:blender", 0),
        )
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**31)

    def test_single_draft_specs_request_one_object_with_kind_specific_guardrails(self) -> None:
        entries = [
            {
                "label": "object:original:window",
                "entry_type": "object",
                "object_name_zh": "窗户",
                "task": "window_care",
                "detailed_actions": ["打开窗户"],
            },
            {
                "label": "scene:MDC-009",
                "entry_type": "scene",
                "command": "帮我把垃圾分类处理",
                "objects": [{"object_name_zh": "垃圾桶"}],
            },
        ]

        specs = gen.build_single_draft_specs(
            entries,
            labels=["object:original:window", "scene:MDC-009"],
        )

        self.assertEqual([spec.kind for spec in specs], ["object", "scene"])
        for spec in specs:
            self.assertIn("严格 JSON object", spec.prompt)
            self.assertIn("只写用户口语命令", spec.prompt)
            self.assertIn("不输出细分动作序列", spec.prompt)
            self.assertIn("不要 JSON 数组", spec.prompt)
            self.assertNotIn("返回严格 JSON 数组", spec.prompt)
        self.assertIn("真实物体能力", specs[0].prompt)
        self.assertIn("整体生活目标", specs[1].prompt)

    def test_single_draft_prompts_require_six_field_self_check_through_rationale(self) -> None:
        entries = [
            {
                "label": "object:original:window",
                "entry_type": "object",
                "object_name_zh": "窗户",
                "task": "window_care",
                "detailed_actions": ["打开窗户"],
            },
            {
                "label": "scene:MDC-009",
                "entry_type": "scene",
                "command": "帮我把垃圾分类处理",
                "objects": [{"object_name_zh": "垃圾桶"}],
            },
        ]

        specs = gen.build_single_draft_specs(
            entries,
            labels=["object:original:window", "scene:MDC-009"],
        )

        for spec in specs:
            self.assertIn("输出前逐字段检查", spec.prompt)
            self.assertIn(
                "text、label_id、expected_executable、object_mentions、operation_evidence、semantic_rationale",
                spec.prompt,
            )
            self.assertIn("不得在 semantic_rationale 之前结束", spec.prompt)
            self.assertIn("不能把多个操作步骤拼成一条", spec.prompt)

    def test_single_draft_mode_rejects_safety_specs_with_handoff(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "independent safety annotation task",
        ):
            gen.build_single_draft_specs([], labels=["safety:cancel_negation"])

    def test_single_draft_parser_accepts_object_and_rejects_array_as_one_draft(self) -> None:
        item = {
            "text": "把窗户打开通通风",
            "label_id": "object:original:window",
            "expected_executable": True,
            "object_mentions": ["窗户"],
            "operation_evidence": ["打开"],
            "semantic_rationale": "明确要求开窗",
        }
        kwargs = {
            "label": "object:original:window",
            "kind": "object",
            "draft_index": 7,
            "seed": 123,
            "prompt_sha256": "p" * 64,
            "generation_parameters": {"temperature": 0.7},
            "model_revisions": {"hf": "h" * 40, "modelscope": "m" * 40},
            "known_labels": {"object:original:window"},
        }

        parsed = gen.parse_single_draft_response(
            json.dumps(item, ensure_ascii=False),
            **kwargs,
        )
        rejected = gen.parse_single_draft_response(
            json.dumps([item], ensure_ascii=False),
            **kwargs,
        )

        self.assertEqual(parsed["parse_status"], "parsed")
        self.assertEqual(parsed["draft_index"], 7)
        self.assertEqual(rejected["parse_status"], "rejected")
        self.assertIn("top_level_must_be_object", rejected["rejection_reasons"])

    def test_single_draft_parser_requires_every_nfkc_span_to_be_contiguous(self) -> None:
        raw = json.dumps(
            {
                "text": "把窗户打开",
                "label_id": "object:original:window",
                "expected_executable": True,
                "object_mentions": ["窗户", "窗-户", "门"],
                "operation_evidence": ["打开", "打 开", "关闭"],
                "semantic_rationale": "明确开窗",
            },
            ensure_ascii=False,
        )

        record = gen.parse_single_draft_response(
            raw,
            label="object:original:window",
            kind="object",
            draft_index=0,
            seed=1,
            prompt_sha256="p" * 64,
            generation_parameters={},
            model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
            known_labels={"object:original:window"},
        )

        self.assertEqual(record["parse_status"], "rejected")
        self.assertIn("object_span_not_contiguous:窗-户", record["rejection_reasons"])
        self.assertIn("object_span_not_contiguous:门", record["rejection_reasons"])
        self.assertIn("operation_span_not_contiguous:打 开", record["rejection_reasons"])
        self.assertIn("operation_span_not_contiguous:关闭", record["rejection_reasons"])

    def test_single_draft_pack_isolates_one_rejection_and_continues_calls(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "窗户",
            "task": "window_care",
            "detailed_actions": ["打开窗户"],
        }
        valid = {
            "text": "把窗户打开通通风",
            "label_id": entry["label"],
            "expected_executable": True,
            "object_mentions": ["窗户"],
            "operation_evidence": ["打开"],
            "semantic_rationale": "明确开窗命令",
        }

        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def generate(self, prompt: str, **kwargs: object) -> str:
                seed = int(kwargs["seed"])
                self.calls.append((prompt, seed))
                if len(self.calls) == 2:
                    return "not json"
                item = dict(valid)
                item["text"] = f"把窗户打开通风{len(self.calls)}分钟"
                item["semantic_rationale"] = f"第{len(self.calls)}条独立草稿"
                return json.dumps(item, ensure_ascii=False)

        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=gen.SINGLE_DRAFT_OUTPUT_DIR) as temporary:
            client = FakeClient()
            manifest = gen.run_single_draft_pack(
                entries=[entry],
                client=client,
                output_dir=Path(temporary),
                labels=[entry["label"]],
                drafts_per_label=3,
                base_seed=314159,
                generation_parameters={"temperature": 0.7},
                model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
                input_hashes={"catalog": "c" * 64, "guidelines": "g" * 64},
                model_manifest_sha256="m" * 64,
                script_sha256="s" * 64,
            )

            self.assertEqual(len(client.calls), 3)
            self.assertEqual(len({seed for _, seed in client.calls}), 3)
            self.assertEqual(manifest["completed_calls"], 3)
            self.assertEqual(manifest["parsed_count"], 2)
            self.assertEqual(manifest["rejected_count"], 1)
            self.assertEqual(manifest["labels"][0]["requested"], 3)
            self.assertEqual(len(list((Path(temporary) / "raw").glob("*.json"))), 3)
            self.assertTrue((Path(temporary) / "draft_manifest.json").is_file())

    def test_single_draft_raw_is_immutable_and_resumes_without_generation(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "窗户",
            "task": "window_care",
            "detailed_actions": ["打开窗户"],
        }
        response = json.dumps(
            {
                "text": "请把窗户关上",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["窗户"],
                "operation_evidence": ["关上"],
                "semantic_rationale": "明确关窗命令",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, **kwargs: object) -> str:
                self.calls += 1
                return response

        kwargs = {
            "entries": [entry],
            "labels": [entry["label"]],
            "drafts_per_label": 1,
            "base_seed": 314159,
            "generation_parameters": {"temperature": 0.7},
            "model_revisions": {"hf": "h" * 40, "modelscope": "m" * 40},
            "input_hashes": {"catalog": "c" * 64, "guidelines": "g" * 64},
            "model_manifest_sha256": "m" * 64,
            "script_sha256": "s" * 64,
        }
        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=gen.SINGLE_DRAFT_OUTPUT_DIR) as temporary:
            client = FakeClient()
            output_dir = Path(temporary)
            first = gen.run_single_draft_pack(
                client=client,
                output_dir=output_dir,
                **kwargs,
            )
            raw_path = output_dir / first["calls"][0]["raw_path"]
            original_bytes = raw_path.read_bytes()
            second = gen.run_single_draft_pack(
                client=client,
                output_dir=output_dir,
                **kwargs,
            )

            self.assertEqual(client.calls, 1)
            self.assertEqual(raw_path.read_bytes(), original_bytes)
            self.assertTrue(second["calls"][0]["resumed"])

    def test_single_draft_resume_hard_fails_on_raw_or_parsed_tampering(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "窗户",
            "task": "window_care",
            "detailed_actions": ["打开窗户"],
        }
        response = json.dumps(
            {
                "text": "把窗户打开",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["窗户"],
                "operation_evidence": ["打开"],
                "semantic_rationale": "明确开窗",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, **kwargs: object) -> str:
                self.calls += 1
                return response

        shared = {
            "entries": [entry],
            "labels": [entry["label"]],
            "drafts_per_label": 1,
            "base_seed": 314159,
            "generation_parameters": {"temperature": 0.7},
            "model_revisions": {"hf": "h" * 40, "modelscope": "m" * 40},
            "input_hashes": {"catalog": "c" * 64, "guidelines": "g" * 64},
            "model_manifest_sha256": "m" * 64,
            "script_sha256": "s" * 64,
        }
        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for artifact_kind in ("raw", "parsed"):
            with self.subTest(artifact_kind=artifact_kind), tempfile.TemporaryDirectory(
                dir=gen.SINGLE_DRAFT_OUTPUT_DIR
            ) as temporary:
                client = FakeClient()
                output_dir = Path(temporary)
                first = gen.run_single_draft_pack(
                    client=client,
                    output_dir=output_dir,
                    **shared,
                )
                path_key = "raw_path" if artifact_kind == "raw" else "parsed_path"
                artifact_path = output_dir / first["calls"][0][path_key]
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                if artifact_kind == "raw":
                    payload["seed"] += 1
                else:
                    payload["semantic_rationale"] = "篡改"
                artifact_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    gen.run_single_draft_pack(
                        client=client,
                        output_dir=output_dir,
                        **shared,
                    )
                self.assertEqual(client.calls, 1)

    def test_single_draft_resume_rejects_full_generation_provenance_matrix(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "窗户",
            "task": "window_care",
            "detailed_actions": ["打开窗户"],
        }
        response = json.dumps(
            {
                "text": "把窗户打开",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["窗户"],
                "operation_evidence": ["打开"],
                "semantic_rationale": "明确开窗",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, **kwargs: object) -> str:
                self.calls += 1
                return response

        base = {
            "entries": [entry],
            "labels": [entry["label"]],
            "drafts_per_label": 1,
            "base_seed": 314159,
            "generation_parameters": {"temperature": 0.7},
            "model_revisions": {"hf": "h" * 40, "modelscope": "m" * 40},
            "input_hashes": {"catalog": "c" * 64, "guidelines": "g" * 64},
            "model_manifest_sha256": "m" * 64,
            "script_sha256": "s" * 64,
        }

        def changed(case: str) -> dict[str, object]:
            value = json.loads(json.dumps(base))
            if case == "prompt":
                value["entries"][0]["detailed_actions"] = ["关闭窗户"]
            elif case == "base_seed":
                value["base_seed"] = 271828
            elif case == "catalog_sha":
                value["input_hashes"]["catalog"] = "x" * 64
            elif case == "guidelines_sha":
                value["input_hashes"]["guidelines"] = "y" * 64
            elif case == "model_manifest_sha":
                value["model_manifest_sha256"] = "z" * 64
            elif case == "hf_revision":
                value["model_revisions"]["hf"] = "x" * 40
            elif case == "modelscope_revision":
                value["model_revisions"]["modelscope"] = "y" * 40
            elif case == "generation_parameters":
                value["generation_parameters"]["temperature"] = 0.8
            elif case == "generated_script_sha":
                value["script_sha256"] = "q" * 64
            return value

        cases = (
            "prompt",
            "base_seed",
            "catalog_sha",
            "guidelines_sha",
            "model_manifest_sha",
            "hf_revision",
            "modelscope_revision",
            "generation_parameters",
            "generated_script_sha",
        )
        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                dir=gen.SINGLE_DRAFT_OUTPUT_DIR
            ) as temporary:
                output_dir = Path(temporary)
                client = FakeClient()
                gen.run_single_draft_pack(client=client, output_dir=output_dir, **base)
                before = {
                    path.relative_to(output_dir).as_posix(): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }

                with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                    gen.run_single_draft_pack(
                        client=client,
                        output_dir=output_dir,
                        **changed(case),
                    )

                after = {
                    path.relative_to(output_dir).as_posix(): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }
                self.assertEqual(client.calls, 1)
                self.assertEqual(after, before)
                resumed = gen.run_single_draft_pack(
                    client=client,
                    output_dir=output_dir,
                    **base,
                )
                self.assertTrue(resumed["calls"][0]["resumed"])
                self.assertEqual(client.calls, 1)
                self.assertEqual(
                    {
                        path.relative_to(output_dir).as_posix(): path.read_bytes()
                        for path in output_dir.rglob("*")
                        if path.is_file()
                    },
                    before,
                )

    def test_single_draft_resume_rejects_linked_artifact_and_manifest_tampering(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "窗户",
            "task": "window_care",
            "detailed_actions": ["打开窗户"],
        }
        response = json.dumps(
            {
                "text": "把窗户打开",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["窗户"],
                "operation_evidence": ["打开"],
                "semantic_rationale": "明确开窗",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, **kwargs: object) -> str:
                self.calls += 1
                return response

        shared = {
            "entries": [entry],
            "labels": [entry["label"]],
            "drafts_per_label": 1,
            "base_seed": 314159,
            "generation_parameters": {"temperature": 0.7},
            "model_revisions": {"hf": "h" * 40, "modelscope": "m" * 40},
            "input_hashes": {"catalog": "c" * 64, "guidelines": "g" * 64},
            "model_manifest_sha256": "m" * 64,
            "script_sha256": "s" * 64,
        }

        def write_json(path: Path, payload: object) -> None:
            path.write_text(gen._json_artifact_text(payload), encoding="utf-8")

        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for case in (
            "raw_filename_linked",
            "parsed_sha_linked",
            "alternate_identity_branch",
            "sidecar",
            "manifest_generated_sha",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                dir=gen.SINGLE_DRAFT_OUTPUT_DIR
            ) as temporary:
                output_dir = Path(temporary)
                client = FakeClient()
                first = gen.run_single_draft_pack(
                    client=client,
                    output_dir=output_dir,
                    **shared,
                )
                call = first["calls"][0]
                raw_path = output_dir / call["raw_path"]
                parsed_path = output_dir / call["parsed_path"]
                sidecar_path = output_dir / call["provenance_path"]
                manifest_path = output_dir / "draft_manifest.json"

                if case == "raw_filename_linked":
                    raw = json.loads(raw_path.read_text(encoding="utf-8"))
                    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    raw["raw_response"] += " "
                    raw["raw_response_sha256"] = gen.sha256_text(raw["raw_response"])
                    parsed["raw_response_sha256"] = raw["raw_response_sha256"]
                    parsed_text = gen._json_artifact_text(parsed)
                    raw["parsed_artifact_sha256"] = gen.sha256_text(parsed_text)
                    new_name = raw_path.name.replace(
                        call["raw_response_sha256"], raw["raw_response_sha256"]
                    )
                    new_raw = raw_path.with_name(new_name)
                    new_parsed = parsed_path.with_name(new_name)
                    raw_path.rename(new_raw)
                    parsed_path.rename(new_parsed)
                    write_json(new_raw, raw)
                    write_json(new_parsed, parsed)
                    sidecar.update(
                        {
                            "raw_response_sha256": raw["raw_response_sha256"],
                            "parsed_artifact_sha256": raw["parsed_artifact_sha256"],
                            "raw_path": new_raw.relative_to(output_dir).as_posix(),
                            "raw_file_sha256": gen.sha256_file(new_raw),
                            "parsed_path": new_parsed.relative_to(output_dir).as_posix(),
                            "parsed_file_sha256": gen.sha256_file(new_parsed),
                        }
                    )
                    write_json(sidecar_path, sidecar)
                elif case == "parsed_sha_linked":
                    raw = json.loads(raw_path.read_text(encoding="utf-8"))
                    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    parsed["semantic_rationale"] = "联动篡改"
                    write_json(parsed_path, parsed)
                    raw["parsed_artifact_sha256"] = gen.sha256_file(parsed_path)
                    write_json(raw_path, raw)
                    sidecar["parsed_artifact_sha256"] = raw["parsed_artifact_sha256"]
                    sidecar["raw_file_sha256"] = gen.sha256_file(raw_path)
                    sidecar["parsed_file_sha256"] = gen.sha256_file(parsed_path)
                    write_json(sidecar_path, sidecar)
                elif case == "alternate_identity_branch":
                    alternate_raw = raw_path.with_name("alternate-" + raw_path.name)
                    alternate_parsed = parsed_path.with_name("alternate-" + parsed_path.name)
                    alternate_raw.write_bytes(raw_path.read_bytes())
                    alternate_parsed.write_bytes(parsed_path.read_bytes())
                elif case == "sidecar":
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    sidecar["untrusted_extra"] = True
                    write_json(sidecar_path, sidecar)
                else:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["generated_script_sha256"] = "x" * 64
                    write_json(manifest_path, manifest)

                before = {
                    path.relative_to(output_dir).as_posix(): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }
                with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                    gen.run_single_draft_pack(
                        client=client,
                        output_dir=output_dir,
                        **shared,
                    )
                self.assertEqual(client.calls, 1)
                self.assertEqual(
                    {
                        path.relative_to(output_dir).as_posix(): path.read_bytes()
                        for path in output_dir.rglob("*")
                        if path.is_file()
                    },
                    before,
                )

    def test_single_draft_resume_rejects_reviewers_five_coordinated_bypasses(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "窗户",
            "task": "window_care",
            "detailed_actions": ["打开窗户"],
        }
        response = json.dumps(
            {
                "text": "把窗户打开",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["窗户"],
                "operation_evidence": ["打开"],
                "semantic_rationale": "明确开窗",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, **kwargs: object) -> str:
                self.calls += 1
                return response

        shared = {
            "entries": [entry],
            "labels": [entry["label"]],
            "drafts_per_label": 1,
            "base_seed": 314159,
            "generation_parameters": {"temperature": 0.7},
            "model_revisions": {"hf": "h" * 40, "modelscope": "m" * 40},
            "input_hashes": {"catalog": "c" * 64, "guidelines": "g" * 64},
            "model_manifest_sha256": "m" * 64,
            "script_sha256": "s" * 64,
        }
        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for case in (
            "delete_manifest_call",
            "manifest_status_reasons",
            "sidecar_input_sha",
            "sidecar_response_sha",
            "sidecar_extra_field",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                dir=gen.SINGLE_DRAFT_OUTPUT_DIR
            ) as temporary:
                output_dir = Path(temporary)
                client = FakeClient()
                first = gen.run_single_draft_pack(
                    client=client, output_dir=output_dir, **shared
                )
                original = {
                    path.relative_to(output_dir).as_posix(): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }
                manifest_path = output_dir / "draft_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                call = manifest["calls"][0]
                sidecar_path = output_dir / call["provenance_path"]
                if case == "delete_manifest_call":
                    manifest["calls"] = []
                elif case == "manifest_status_reasons":
                    call["parse_status"] = "rejected"
                    call["rejection_reasons"] = ["coordinated_tamper"]
                else:
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    if case == "sidecar_input_sha":
                        sidecar["input_sha256"]["catalog"] = "x" * 64
                    elif case == "sidecar_response_sha":
                        sidecar["raw_response_sha256"] = "r" * 64
                        call["raw_response_sha256"] = "r" * 64
                    else:
                        sidecar["untrusted_extra"] = True
                    sidecar_path.write_text(
                        gen._json_artifact_text(sidecar), encoding="utf-8"
                    )
                    call["provenance_file_sha256"] = gen.sha256_file(sidecar_path)
                manifest_path.write_text(
                    gen._json_artifact_text(manifest), encoding="utf-8"
                )
                gen._write_completion_record(
                    output_dir,
                    first["generation_run_sha256"],
                    {(entry["label"], 0)},
                )
                tampered = {
                    path.relative_to(output_dir).as_posix(): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }

                with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                    gen.run_single_draft_pack(
                        client=client, output_dir=output_dir, **shared
                    )
                self.assertEqual(client.calls, 1)
                self.assertEqual(
                    {
                        path.relative_to(output_dir).as_posix(): path.read_bytes()
                        for path in output_dir.rglob("*")
                        if path.is_file()
                    },
                    tampered,
                )
                for path in output_dir.rglob("*"):
                    if (
                        path.is_file()
                        and path.relative_to(output_dir).as_posix() not in original
                    ):
                        path.unlink()
                for relative, contents in original.items():
                    (output_dir / relative).write_bytes(contents)
                resumed = gen.run_single_draft_pack(
                    client=client, output_dir=output_dir, **shared
                )
                self.assertTrue(resumed["calls"][0]["resumed"])
                self.assertEqual(client.calls, 1)
                self.assertEqual(
                    {
                        path.relative_to(output_dir).as_posix(): path.read_bytes()
                        for path in output_dir.rglob("*")
                        if path.is_file()
                    },
                    original,
                )

    def test_single_draft_migration_records_superseded_hashes_and_dual_script_sha(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "窗户",
            "task": "window_care",
            "detailed_actions": ["打开窗户"],
        }
        response = json.dumps(
            {
                "text": "把窗户打开",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["窗户"],
                "operation_evidence": ["打开"],
                "semantic_rationale": "明确开窗",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def generate(self, prompt: str, **kwargs: object) -> str:
                return response

        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=gen.SINGLE_DRAFT_OUTPUT_DIR) as temporary:
            output_dir = Path(temporary)
            manifest = gen.run_single_draft_pack(
                entries=[entry],
                client=FakeClient(),
                output_dir=output_dir,
                labels=[entry["label"]],
                drafts_per_label=1,
                base_seed=314159,
                generation_parameters={"temperature": 0.7},
                model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
                input_hashes={"catalog": "c" * 64, "guidelines": "g" * 64},
                model_manifest_sha256="m" * 64,
                script_sha256="s" * 64,
            )
            formal_raw = output_dir / manifest["calls"][0]["raw_path"]
            formal_parsed = output_dir / manifest["calls"][0]["parsed_path"]
            old_raw = formal_raw.with_name("old-" + formal_raw.name)
            old_parsed = formal_parsed.with_name("old-" + formal_parsed.name)
            old_raw.write_bytes(formal_raw.read_bytes())
            old_parsed.write_bytes(formal_parsed.read_bytes())
            old_raw_sha = gen.sha256_file(old_raw)
            old_parsed_sha = gen.sha256_file(old_parsed)
            old_manifest_path = output_dir / "draft_manifest.json"
            old_manifest_bytes = old_manifest_path.read_bytes()
            old_manifest_sha = gen.sha256_file(old_manifest_path)

            migrated = gen.migrate_single_draft_artifacts(
                output_dir,
                delivered_script_sha256="d" * 64,
            )

            self.assertEqual(migrated["manifest_version"], 2)
            self.assertEqual(migrated["generated_script_sha256"], "s" * 64)
            self.assertEqual(migrated["delivered_script_sha256"], "d" * 64)
            self.assertEqual(
                migrated["migration"]["previous_manifest_sha256"],
                old_manifest_sha,
            )
            previous_snapshot = output_dir / migrated["migration"][
                "previous_manifest_snapshot_path"
            ]
            self.assertEqual(previous_snapshot.read_bytes(), old_manifest_bytes)
            self.assertEqual(gen.sha256_file(previous_snapshot), old_manifest_sha)
            self.assertTrue((output_dir / "migration" / "plan.json").is_file())
            state = json.loads(
                (output_dir / "migration" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "complete")
            self.assertEqual(
                migrated["parser_migration_audit"],
                {
                    "formal_artifact_count": 1,
                    "changed_record_count": 0,
                    "delivered_parser_parsed": 1,
                    "delivered_parser_rejected": 0,
                    "formal_artifacts_rewritten": False,
                },
            )
            audit = migrated["superseded_audit"]
            self.assertEqual(audit["directory"], "superseded")
            self.assertEqual(audit["raw_count"], 1)
            self.assertEqual(audit["parsed_count"], 1)
            self.assertEqual(audit["pairs"][0]["raw_sha256"], old_raw_sha)
            self.assertEqual(audit["pairs"][0]["parsed_sha256"], old_parsed_sha)
            migrated_raw = output_dir / audit["pairs"][0]["migrated_raw_path"]
            migrated_parsed = output_dir / audit["pairs"][0]["migrated_parsed_path"]
            self.assertEqual(gen.sha256_file(migrated_raw), old_raw_sha)
            self.assertEqual(gen.sha256_file(migrated_parsed), old_parsed_sha)
            self.assertFalse(old_raw.exists())
            self.assertFalse(old_parsed.exists())
            self.assertTrue((output_dir / migrated["generation_run_path"]).is_file())
            self.assertTrue(
                (output_dir / migrated["calls"][0]["provenance_path"]).is_file()
            )
            self.assertEqual(len(list((output_dir / "raw").glob("*.json"))), 1)
            self.assertEqual(len(list((output_dir / "parsed").glob("*.json"))), 1)
            second = gen.migrate_single_draft_artifacts(
                output_dir,
                delivered_script_sha256="d" * 64,
            )
            self.assertEqual(second["superseded_audit"], audit)
            self.assertEqual(second["migration"], migrated["migration"])

    def test_single_draft_resume_requires_exact_call_and_artifact_schemas(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "绐楁埛",
            "task": "window_care",
            "detailed_actions": ["鎵撳紑绐楁埛"],
        }
        response = json.dumps(
            {
                "text": "鎶婄獥鎴锋墦寮€",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["绐楁埛"],
                "operation_evidence": ["鎵撳紑"],
                "semantic_rationale": "鏄庣‘寮€绐?",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, **kwargs: object) -> str:
                self.calls += 1
                return response

        shared = {
            "entries": [entry],
            "labels": [entry["label"]],
            "drafts_per_label": 1,
            "base_seed": 314159,
            "generation_parameters": {"temperature": 0.7},
            "model_revisions": {"hf": "h" * 40, "modelscope": "m" * 40},
            "input_hashes": {"catalog": "c" * 64, "guidelines": "g" * 64},
            "model_manifest_sha256": "m" * 64,
            "script_sha256": "s" * 64,
        }
        cases = (
            "missing_call",
            "duplicate_call",
            "extra_call",
            "manifest_extra",
            "call_extra",
            "sidecar_extra",
            "raw_extra",
            "parsed_extra",
        )
        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                dir=gen.SINGLE_DRAFT_OUTPUT_DIR
            ) as temporary:
                output_dir = Path(temporary)
                client = FakeClient()
                first = gen.run_single_draft_pack(
                    client=client, output_dir=output_dir, **shared
                )
                manifest_path = output_dir / "draft_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                call = manifest["calls"][0]
                raw_path = output_dir / call["raw_path"]
                parsed_path = output_dir / call["parsed_path"]
                sidecar_path = output_dir / call["provenance_path"]
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if case == "missing_call":
                    manifest["calls"] = []
                elif case == "duplicate_call":
                    manifest["calls"].append(dict(call))
                elif case == "extra_call":
                    extra = dict(call)
                    extra["draft_index"] = 99
                    manifest["calls"].append(extra)
                elif case == "manifest_extra":
                    manifest["untrusted_extra"] = True
                elif case == "call_extra":
                    call["untrusted_extra"] = True
                elif case == "sidecar_extra":
                    sidecar["untrusted_extra"] = True
                    sidecar_path.write_text(
                        gen._json_artifact_text(sidecar), encoding="utf-8"
                    )
                    call["provenance_file_sha256"] = gen.sha256_file(sidecar_path)
                elif case == "raw_extra":
                    raw = json.loads(raw_path.read_text(encoding="utf-8"))
                    raw["untrusted_extra"] = True
                    raw_path.write_text(gen._json_artifact_text(raw), encoding="utf-8")
                    sidecar["raw_file_sha256"] = gen.sha256_file(raw_path)
                    sidecar_path.write_text(
                        gen._json_artifact_text(sidecar), encoding="utf-8"
                    )
                    call["raw_file_sha256"] = sidecar["raw_file_sha256"]
                    call["provenance_file_sha256"] = gen.sha256_file(sidecar_path)
                else:
                    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
                    parsed["untrusted_extra"] = True
                    parsed_path.write_text(
                        gen._json_artifact_text(parsed), encoding="utf-8"
                    )
                    parsed_sha = gen.sha256_file(parsed_path)
                    raw = json.loads(raw_path.read_text(encoding="utf-8"))
                    raw["parsed_artifact_sha256"] = parsed_sha
                    raw_path.write_text(gen._json_artifact_text(raw), encoding="utf-8")
                    sidecar.update(
                        {
                            "parsed_artifact_sha256": parsed_sha,
                            "raw_file_sha256": gen.sha256_file(raw_path),
                            "parsed_file_sha256": gen.sha256_file(parsed_path),
                        }
                    )
                    sidecar_path.write_text(
                        gen._json_artifact_text(sidecar), encoding="utf-8"
                    )
                    call.update(
                        {
                            "parsed_artifact_sha256": parsed_sha,
                            "raw_file_sha256": sidecar["raw_file_sha256"],
                            "parsed_file_sha256": sidecar["parsed_file_sha256"],
                            "provenance_file_sha256": gen.sha256_file(sidecar_path),
                        }
                    )
                manifest_path.write_text(
                    gen._json_artifact_text(manifest), encoding="utf-8"
                )
                gen._write_completion_record(
                    output_dir,
                    first["generation_run_sha256"],
                    {(entry["label"], 0)},
                )
                before = {
                    path.relative_to(output_dir).as_posix(): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }
                with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                    gen.run_single_draft_pack(
                        client=client, output_dir=output_dir, **shared
                    )
                self.assertEqual(client.calls, 1)
                self.assertEqual(
                    {
                        path.relative_to(output_dir).as_posix(): path.read_bytes()
                        for path in output_dir.rglob("*")
                        if path.is_file()
                    },
                    before,
                )

    def test_single_draft_resume_rejects_coordinated_manifest_summary_tampering(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "绐楁埛",
            "task": "window_care",
            "detailed_actions": ["鎵撳紑绐楁埛"],
        }
        response = json.dumps(
            {
                "text": "鎶婄獥鎴锋墦寮€",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["绐楁埛"],
                "operation_evidence": ["鎵撳紑"],
                "semantic_rationale": "鏄庣‘寮€绐?",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, **kwargs: object) -> str:
                self.calls += 1
                return response

        shared = {
            "entries": [entry],
            "labels": [entry["label"]],
            "drafts_per_label": 1,
            "base_seed": 314159,
            "generation_parameters": {"temperature": 0.7},
            "model_revisions": {"hf": "h" * 40, "modelscope": "m" * 40},
            "input_hashes": {"catalog": "c" * 64, "guidelines": "g" * 64},
            "model_manifest_sha256": "m" * 64,
            "script_sha256": "s" * 64,
        }
        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for case in (
            "status",
            "counts",
            "label_summary",
            "delivered_script",
            "generation_run_path",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                dir=gen.SINGLE_DRAFT_OUTPUT_DIR
            ) as temporary:
                output_dir = Path(temporary)
                client = FakeClient()
                first = gen.run_single_draft_pack(
                    client=client, output_dir=output_dir, **shared
                )
                manifest_path = output_dir / "draft_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if case == "status":
                    manifest["status"] = "partial"
                elif case == "counts":
                    manifest["completed_calls"] = 0
                    manifest["parsed_count"] = 0
                    manifest["rejected_count"] = 0
                elif case == "label_summary":
                    manifest["labels"][0]["parsed"] += 1
                elif case == "delivered_script":
                    manifest["delivered_script_sha256"] = "d" * 64
                else:
                    manifest["generation_run_path"] = "alternate-run.json"
                manifest_path.write_text(
                    gen._json_artifact_text(manifest), encoding="utf-8"
                )
                gen._write_completion_record(
                    output_dir,
                    first["generation_run_sha256"],
                    {(entry["label"], 0)},
                )
                before = {
                    path.relative_to(output_dir).as_posix(): path.read_bytes()
                    for path in output_dir.rglob("*")
                    if path.is_file()
                }
                with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                    gen.run_single_draft_pack(
                        client=client, output_dir=output_dir, **shared
                    )
                self.assertEqual(client.calls, 1)
                self.assertEqual(
                    {
                        path.relative_to(output_dir).as_posix(): path.read_bytes()
                        for path in output_dir.rglob("*")
                        if path.is_file()
                    },
                    before,
                )

    def test_single_draft_migration_recovers_after_interrupted_pair_move(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "绐楁埛",
            "task": "window_care",
            "detailed_actions": ["鎵撳紑绐楁埛"],
        }
        response = json.dumps(
            {
                "text": "鎶婄獥鎴锋墦寮€",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["绐楁埛"],
                "operation_evidence": ["鎵撳紑"],
                "semantic_rationale": "鏄庣‘寮€绐?",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def generate(self, prompt: str, **kwargs: object) -> str:
                return response

        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=gen.SINGLE_DRAFT_OUTPUT_DIR) as temporary:
            output_dir = Path(temporary)
            manifest = gen.run_single_draft_pack(
                entries=[entry],
                client=FakeClient(),
                output_dir=output_dir,
                labels=[entry["label"]],
                drafts_per_label=1,
                base_seed=314159,
                generation_parameters={"temperature": 0.7},
                model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
                input_hashes={"catalog": "c" * 64, "guidelines": "g" * 64},
                model_manifest_sha256="m" * 64,
                script_sha256="s" * 64,
            )
            formal_raw = output_dir / manifest["calls"][0]["raw_path"]
            formal_parsed = output_dir / manifest["calls"][0]["parsed_path"]
            for prefix in ("old-a-", "old-b-"):
                formal_raw.with_name(prefix + formal_raw.name).write_bytes(
                    formal_raw.read_bytes()
                )
                formal_parsed.with_name(prefix + formal_parsed.name).write_bytes(
                    formal_parsed.read_bytes()
                )
            old_manifest_bytes = (output_dir / "draft_manifest.json").read_bytes()
            completed: list[str] = []

            def interrupt_after_first(pair_name: str) -> None:
                completed.append(pair_name)
                if len(completed) == 1:
                    raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                gen.migrate_single_draft_artifacts(
                    output_dir,
                    delivered_script_sha256="d" * 64,
                    progress_hook=interrupt_after_first,
                )

            self.assertEqual(
                (output_dir / "draft_manifest.json").read_bytes(), old_manifest_bytes
            )
            self.assertTrue((output_dir / "migration" / "plan.json").is_file())
            interrupted_state = json.loads(
                (output_dir / "migration" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(interrupted_state["status"], "moving")
            self.assertEqual(len(interrupted_state["completed_pairs"]), 1)
            snapshot_path = output_dir / interrupted_state["previous_manifest_snapshot_path"]
            self.assertEqual(snapshot_path.read_bytes(), old_manifest_bytes)

            migrated = gen.migrate_single_draft_artifacts(
                output_dir, delivered_script_sha256="d" * 64
            )
            self.assertEqual(len(migrated["superseded_audit"]["pairs"]), 2)
            final_state = json.loads(
                (output_dir / "migration" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final_state["status"], "complete")
            self.assertEqual(len(final_state["completed_pairs"]), 2)

    def test_single_draft_migration_rejects_absolute_and_relative_path_escape(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "绐楁埛",
            "task": "window_care",
            "detailed_actions": ["鎵撳紑绐楁埛"],
        }
        response = json.dumps(
            {
                "text": "鎶婄獥鎴锋墦寮€",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["绐楁埛"],
                "operation_evidence": ["鎵撳紑"],
                "semantic_rationale": "鏄庣‘寮€绐?",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def generate(self, prompt: str, **kwargs: object) -> str:
                return response

        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for case in (
            "absolute",
            "dotdot",
            "wrong_directory",
            "name_mismatch",
            "duplicate_pair",
            "pair_extra_field",
            "state_extra_field",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                dir=gen.SINGLE_DRAFT_OUTPUT_DIR
            ) as temporary, tempfile.TemporaryDirectory(
                dir=gen.SINGLE_DRAFT_OUTPUT_DIR
            ) as escape_temporary:
                output_dir = Path(temporary)
                escape_dir = Path(escape_temporary)
                manifest = gen.run_single_draft_pack(
                    entries=[entry],
                    client=FakeClient(),
                    output_dir=output_dir,
                    labels=[entry["label"]],
                    drafts_per_label=1,
                    base_seed=314159,
                    generation_parameters={"temperature": 0.7},
                    model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
                    input_hashes={"catalog": "c" * 64, "guidelines": "g" * 64},
                    model_manifest_sha256="m" * 64,
                    script_sha256="s" * 64,
                )
                formal_raw = output_dir / manifest["calls"][0]["raw_path"]
                formal_parsed = output_dir / manifest["calls"][0]["parsed_path"]
                extra_raw = formal_raw.with_name("old-" + formal_raw.name)
                extra_parsed = formal_parsed.with_name("old-" + formal_parsed.name)
                extra_raw.write_bytes(formal_raw.read_bytes())
                extra_parsed.write_bytes(formal_parsed.read_bytes())
                migrated = gen.migrate_single_draft_artifacts(
                    output_dir, delivered_script_sha256="d" * 64
                )
                pair = migrated["superseded_audit"]["pairs"][0]
                escape_raw = escape_dir / "escape-raw.json"
                escape_parsed = escape_dir / "escape-parsed.json"
                escape_raw.write_bytes(
                    (output_dir / pair["migrated_raw_path"]).read_bytes()
                )
                escape_parsed.write_bytes(
                    (output_dir / pair["migrated_parsed_path"]).read_bytes()
                )
                target_raw = escape_dir / "moved-raw.json"
                target_parsed = escape_dir / "moved-parsed.json"
                plan_path = output_dir / "migration" / "plan.json"
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                if case == "absolute":
                    paths = [escape_raw, target_raw, escape_parsed, target_parsed]
                    values = [path.as_posix() for path in paths]
                elif case == "dotdot":
                    prefix = f"../{escape_dir.name}"
                    values = [
                        f"{prefix}/escape-raw.json",
                        f"{prefix}/moved-raw.json",
                        f"{prefix}/escape-parsed.json",
                        f"{prefix}/moved-parsed.json",
                    ]
                else:
                    values = []
                if values:
                    plan["pairs"][0].update(
                        {
                            "original_raw_path": values[0],
                            "migrated_raw_path": values[1],
                            "raw_sha256": gen.sha256_file(escape_raw),
                            "original_parsed_path": values[2],
                            "migrated_parsed_path": values[3],
                            "parsed_sha256": gen.sha256_file(escape_parsed),
                        }
                    )
                elif case == "wrong_directory":
                    name = Path(plan["pairs"][0]["original_raw_path"]).name
                    plan["pairs"][0]["original_raw_path"] = f"parsed/{name}"
                elif case == "name_mismatch":
                    plan["pairs"][0]["migrated_raw_path"] = (
                        "superseded/raw/different.json"
                    )
                elif case == "duplicate_pair":
                    plan["pairs"].append(dict(plan["pairs"][0]))
                elif case == "pair_extra_field":
                    plan["pairs"][0]["untrusted_extra"] = True
                plan_path.write_text(gen._json_artifact_text(plan), encoding="utf-8")
                state_path = output_dir / "migration" / "state.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["plan_sha256"] = gen.sha256_file(plan_path)
                if case == "state_extra_field":
                    state["untrusted_extra"] = True
                state_path.write_text(gen._json_artifact_text(state), encoding="utf-8")
                before = {
                    path.relative_to(escape_dir).as_posix(): path.read_bytes()
                    for path in escape_dir.rglob("*")
                    if path.is_file()
                }

                with self.assertRaisesRegex(RuntimeError, "migration|provenance"):
                    gen.migrate_single_draft_artifacts(
                        output_dir, delivered_script_sha256="d" * 64
                    )

                self.assertEqual(
                    {
                        path.relative_to(escape_dir).as_posix(): path.read_bytes()
                        for path in escape_dir.rglob("*")
                        if path.is_file()
                    },
                    before,
                )

    def test_single_draft_migration_rejects_symlinked_artifact_path(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "绐楁埛",
            "task": "window_care",
            "detailed_actions": ["鎵撳紑绐楁埛"],
        }
        response = json.dumps(
            {
                "text": "鎶婄獥鎴锋墦寮€",
                "label_id": entry["label"],
                "expected_executable": True,
                "object_mentions": ["绐楁埛"],
                "operation_evidence": ["鎵撳紑"],
                "semantic_rationale": "鏄庣‘寮€绐?",
            },
            ensure_ascii=False,
        )

        class FakeClient:
            def generate(self, prompt: str, **kwargs: object) -> str:
                return response

        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=gen.SINGLE_DRAFT_OUTPUT_DIR
        ) as temporary, tempfile.TemporaryDirectory(
            dir=gen.SINGLE_DRAFT_OUTPUT_DIR
        ) as external_temporary:
            output_dir = Path(temporary)
            manifest = gen.run_single_draft_pack(
                entries=[entry],
                client=FakeClient(),
                output_dir=output_dir,
                labels=[entry["label"]],
                drafts_per_label=1,
                base_seed=314159,
                generation_parameters={"temperature": 0.7},
                model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
                input_hashes={"catalog": "c" * 64, "guidelines": "g" * 64},
                model_manifest_sha256="m" * 64,
                script_sha256="s" * 64,
            )
            formal_raw = output_dir / manifest["calls"][0]["raw_path"]
            formal_parsed = output_dir / manifest["calls"][0]["parsed_path"]
            extra_raw = formal_raw.with_name("old-" + formal_raw.name)
            extra_parsed = formal_parsed.with_name("old-" + formal_parsed.name)
            extra_raw.write_bytes(formal_raw.read_bytes())
            extra_parsed.write_bytes(formal_parsed.read_bytes())
            migrated = gen.migrate_single_draft_artifacts(
                output_dir, delivered_script_sha256="d" * 64
            )
            pair = migrated["superseded_audit"]["pairs"][0]
            target = output_dir / pair["migrated_raw_path"]
            external = Path(external_temporary) / "external.json"
            external.write_bytes(target.read_bytes())
            target.unlink()
            try:
                target.symlink_to(external)
            except OSError as exc:
                target.write_bytes(external.read_bytes())
                with mock.patch.object(
                    gen,
                    "_path_has_reparse_point",
                    side_effect=lambda path: path == target,
                ):
                    with self.assertRaisesRegex(RuntimeError, "migration path"):
                        gen.migrate_single_draft_artifacts(
                            output_dir, delivered_script_sha256="d" * 64
                        )
                self.assertEqual(external.read_bytes(), target.read_bytes())
                return
            external_before = external.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "migration path"):
                gen.migrate_single_draft_artifacts(
                    output_dir, delivered_script_sha256="d" * 64
                )

            self.assertTrue(target.is_symlink())
            self.assertEqual(external.read_bytes(), external_before)

    def test_single_draft_cli_dry_run_plans_ten_by_thirty_without_model_load(self) -> None:
        loaded = False

        def forbidden_loader(*args: object, **kwargs: object) -> object:
            nonlocal loaded
            loaded = True
            raise AssertionError("dry-run must not load Qwen")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = gen.main(
                ["--single-draft", "--dry-run"],
                model_loader=forbidden_loader,
            )

        plan = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertFalse(loaded)
        self.assertEqual(plan["mode"], "single-draft")
        self.assertEqual(plan["label_count"], 10)
        self.assertEqual(plan["candidate_count"], 300)
        self.assertEqual(plan["output_dir"], str(gen.SINGLE_DRAFT_OUTPUT_DIR))

    def test_single_draft_report_is_draft_only_and_summarizes_each_label(self) -> None:
        manifest = {
            "completed_calls": 2,
            "expected_calls": 2,
            "parsed_count": 1,
            "rejected_count": 1,
            "elapsed_seconds": 1.25,
            "superseded_audit": {
                "raw_count": 62,
                "parsed_count": 62,
                "reason": "initial prompt omitted semantic_rationale",
            },
            "delivered_script_sha256": "d" * 64,
            "migration": {
                "previous_manifest_sha256": "o" * 64,
                "formal_raw_count": 2,
                "formal_parsed_count": 2,
            },
            "parser_migration_audit": {
                "formal_artifact_count": 2,
                "changed_record_count": 1,
                "delivered_parser_parsed": 1,
                "delivered_parser_rejected": 1,
                "formal_artifacts_rewritten": False,
            },
            "labels": [
                {
                    "label": "object:original:window",
                    "kind": "object",
                    "requested": 2,
                    "parsed": 1,
                    "rejected": 1,
                    "elapsed_seconds": 1.0,
                }
            ],
            "model_revisions": {"hf": "h" * 40, "modelscope": "m" * 40},
            "input_sha256": {"catalog": "c" * 64, "guidelines": "g" * 64},
            "model_manifest_sha256": "m" * 64,
            "script_sha256": "s" * 64,
            "calls": [
                {"rejection_reasons": ["missing_direct_operation_span"]}
            ],
        }
        gen.SINGLE_DRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=gen.SINGLE_DRAFT_OUTPUT_DIR) as temporary:
            report_path = Path(temporary) / "report.md"
            gen.write_single_draft_report(
                manifest,
                path=report_path,
                test_summary="Ran 20 tests - OK",
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("DONE_WITH_CONCERNS", report)
        self.assertIn("Ran 20 tests - OK", report)
        self.assertIn("object:original:window", report)
        self.assertIn("2 / 2", report)
        self.assertIn("不可变草稿", report)
        self.assertIn("不签发训练样本", report)
        self.assertIn("Superseded audit", report)
        self.assertIn("62", report)
        self.assertIn("missing_direct_operation_span", report)
        self.assertIn("d" * 64, report)
        self.assertIn("Migration audit", report)
        self.assertIn("o" * 64, report)
        self.assertIn("changed record：1", report)

    def test_candidate_generator_module_exists(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("tools.generate_spoken_annotation_candidates")
        )

    def test_object_prompt_contains_only_target_context_and_role_guardrails(self) -> None:
        entry = {
            "label": "object:original:blender",
            "entry_type": "object",
            "object_name_zh": "搅拌机",
            "task": "smart_cooking",
            "detailed_actions": ["检查容量", "执行搅拌"],
        }

        self.assertTrue(hasattr(gen, "build_object_prompt"))
        prompt = gen.build_object_prompt(entry, 12)

        self.assertIn("object:original:blender", prompt)
        self.assertIn("搅拌机", prompt)
        self.assertIn("固定设施", prompt)
        self.assertIn("真实可操作性", prompt)
        self.assertIn("严格 JSON 数组", prompt)
        self.assertIn("12", prompt)
        self.assertNotIn("scene:", prompt)

    def test_prompts_demand_verbatim_contiguous_span_copying(self) -> None:
        entry = {
            "label": "object:original:window",
            "entry_type": "object",
            "object_name_zh": "窗户",
            "task": "window_care",
            "detailed_actions": ["打开窗户"],
        }

        prompts = [
            gen.build_object_prompt(entry, 12),
            gen.build_scene_prompt(
                {
                    "label": "scene:MDC-009",
                    "command": "处理垃圾",
                    "objects": [{"object_name_zh": "垃圾桶"}],
                },
                12,
            ),
            gen.build_safety_prompt("cancel_negation", 12, [entry]),
        ]

        for prompt in prompts:
            self.assertIn("逐字复制", prompt)
            self.assertIn("连续片段", prompt)
            self.assertIn('"operation_evidence":["打开"]', prompt)

    def test_scene_prompt_requires_whole_goal_without_object_enumeration(self) -> None:
        entry = {
            "label": "scene:MDC-009",
            "entry_type": "scene",
            "command": "帮我把垃圾分类处理",
            "objects": [
                {"object_name_zh": "垃圾桶"},
                {"object_name_zh": "电池"},
            ],
        }

        self.assertTrue(hasattr(gen, "build_scene_prompt"))
        prompt = gen.build_scene_prompt(entry, 12)

        self.assertIn("帮我把垃圾分类处理", prompt)
        self.assertIn("垃圾桶", prompt)
        self.assertIn("整体生活目标", prompt)
        self.assertIn("不机械枚举", prompt)
        self.assertIn("固定设施", prompt)
        self.assertIn("垃圾恢复", prompt)
        self.assertIn("同义对象", prompt)

    def test_safety_mix_scales_guideline_ratio_to_twelve(self) -> None:
        self.assertTrue(hasattr(gen, "safety_mix"))

        self.assertEqual(
            gen.safety_mix("restrictive_negation_or_in_word_boundary", 12),
            (6, 6),
        )
        self.assertEqual(
            gen.safety_mix("multi_clause_turn_correction", 12),
            (6, 6),
        )
        self.assertEqual(gen.safety_mix("cancel_negation", 12), (0, 12))

    def test_safety_prompt_states_class_mix_and_action_free_rejections(self) -> None:
        self.assertTrue(hasattr(gen, "build_safety_prompt"))
        prompt = gen.build_safety_prompt(
            "multi_clause_turn_correction",
            12,
            [{"label": "object:original:window", "object_name_zh": "窗户"}],
        )

        self.assertIn("multi_clause_turn_correction", prompt)
        self.assertIn("6 条 expected_executable=true", prompt)
        self.assertIn("6 条 expected_executable=false", prompt)
        self.assertIn("不得暗含任何可执行动作", prompt)
        self.assertIn("operation_evidence 必须为空数组", prompt)

    def test_safety_prompt_assigns_every_index_and_requires_six_fields_per_item(self) -> None:
        prompt = gen.build_safety_prompt(
            "multi_clause_turn_correction",
            12,
            [{"label": "object:original:window", "object_name_zh": "窗户"}],
        )

        self.assertIn("第 1 至 6 项", prompt)
        self.assertIn("第 7 至 12 项", prompt)
        self.assertIn("每一项都必须包含以下 6 个字段", prompt)
        self.assertIn("数组外文字", prompt)

    def test_parser_accepts_structured_core_candidate_and_preserves_provenance(self) -> None:
        self.assertTrue(hasattr(gen, "parse_batch_response"))
        raw = json.dumps(
            [
                {
                    "text": "把窗户打开通通风",
                    "label_id": "object:original:window",
                    "expected_executable": True,
                    "object_mentions": ["窗户"],
                    "operation_evidence": ["打开"],
                    "semantic_rationale": "明确对象和开窗操作",
                }
            ],
            ensure_ascii=False,
        )

        records = gen.parse_batch_response(
            raw,
            batch_label="object:original:window",
            safety_class=None,
            count=1,
            seed=314159,
            prompt_sha256="p" * 64,
            generation_parameters={"temperature": 0.7},
            model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
            known_labels={"object:original:window"},
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["parse_status"], "parsed")
        self.assertEqual(record["rejection_reasons"], [])
        self.assertEqual(record["label_id"], "object:original:window")
        self.assertEqual(record["raw_response_sha256"], gen.sha256_text(raw))
        self.assertEqual(record["prompt_sha256"], "p" * 64)
        self.assertEqual(record["seed"], 314159)
        self.assertTrue(record["candidate_id"])

    def test_local_loader_verifies_before_strict_offline_bf16_cuda_load(self) -> None:
        self.assertTrue(hasattr(gen, "load_local_qwen"))
        events: list[tuple[str, object]] = []

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def is_bf16_supported() -> bool:
                return True

        class FakeTorch:
            cuda = FakeCuda()
            bfloat16 = "bf16"

        class FakeTokenizerFactory:
            @staticmethod
            def from_pretrained(path: str, **kwargs: object) -> str:
                events.append(("tokenizer", (path, kwargs)))
                return "TOKENIZER"

        class FakeModelFactory:
            @staticmethod
            def from_pretrained(path: str, **kwargs: object) -> str:
                events.append(("model", (path, kwargs)))
                return "MODEL"

        def fake_verify(path: Path) -> dict[str, object]:
            events.append(("verify", path))
            return {"revision": "h" * 40, "transport_revision": "m" * 40}

        tokenizer, model, manifest = gen.load_local_qwen(
            Path("local-model"),
            verifier=fake_verify,
            tokenizer_factory=FakeTokenizerFactory,
            model_factory=FakeModelFactory,
            torch_module=FakeTorch,
        )

        self.assertEqual(tokenizer, "TOKENIZER")
        self.assertEqual(model, "MODEL")
        self.assertEqual(manifest["revision"], "h" * 40)
        self.assertEqual([event[0] for event in events], ["verify", "tokenizer", "model"])
        tokenizer_kwargs = events[1][1][1]
        model_kwargs = events[2][1][1]
        self.assertEqual(
            tokenizer_kwargs,
            {"local_files_only": True, "trust_remote_code": False},
        )
        self.assertTrue(model_kwargs["local_files_only"])
        self.assertFalse(model_kwargs["trust_remote_code"])
        self.assertTrue(model_kwargs["use_safetensors"])
        self.assertEqual(model_kwargs["torch_dtype"], "bf16")
        self.assertEqual(model_kwargs["device_map"], {"": "cuda"})

    def test_qwen_client_uses_chat_template_fixed_seed_and_decodes_completion_only(self) -> None:
        self.assertTrue(hasattr(gen, "QwenLocalClient"))
        events: list[tuple[str, object]] = []

        class FakeTensor:
            def __init__(self, values: list[int]) -> None:
                self.values = values
                self.shape = (1, len(values))

            def to(self, device: str) -> "FakeTensor":
                events.append(("tensor_to", device))
                return self

            def __getitem__(self, item: object) -> object:
                return self.values[item]

        class FakeTokenizer:
            eos_token_id = 99

            def apply_chat_template(self, messages: object, **kwargs: object) -> dict[str, FakeTensor]:
                events.append(("template", (messages, kwargs)))
                return {"input_ids": FakeTensor([10, 11]), "attention_mask": FakeTensor([1, 1])}

            def decode(self, tokens: object, **kwargs: object) -> str:
                events.append(("decode", (tokens, kwargs)))
                return '[{"text":"ok"}]'

        class FakeModel:
            def generate(self, **kwargs: object) -> list[FakeTensor]:
                events.append(("generate", kwargs))
                return [FakeTensor([10, 11, 20, 21])]

        class FakeInference:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *args: object) -> None:
                return None

        class FakeCuda:
            @staticmethod
            def manual_seed_all(seed: int) -> None:
                events.append(("cuda_seed", seed))

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def manual_seed(seed: int) -> None:
                events.append(("seed", seed))

            @staticmethod
            def inference_mode() -> FakeInference:
                return FakeInference()

        client = gen.QwenLocalClient(FakeTokenizer(), FakeModel(), FakeTorch)
        response = client.generate(
            "PROMPT",
            seed=123,
            generation_parameters={
                "max_new_tokens": 128,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

        self.assertEqual(response, '[{"text":"ok"}]')
        self.assertIn(("seed", 123), events)
        self.assertIn(("cuda_seed", 123), events)
        generate_kwargs = next(value for name, value in events if name == "generate")
        self.assertEqual(generate_kwargs["max_new_tokens"], 128)
        decoded_tokens = next(value for name, value in events if name == "decode")[0]
        self.assertEqual(decoded_tokens, [20, 21])

    def test_batch_runner_writes_raw_and_candidates_then_resumes_without_generation(self) -> None:
        self.assertTrue(hasattr(gen, "BatchSpec"))
        self.assertTrue(hasattr(gen, "run_batch"))
        gen.DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, **kwargs: object) -> str:
                self.calls += 1
                return json.dumps(
                    [
                        {
                            "text": "把窗户打开",
                            "label_id": "object:original:window",
                            "expected_executable": True,
                            "object_mentions": ["窗户"],
                            "operation_evidence": ["打开"],
                            "semantic_rationale": "开窗",
                        },
                        {
                            "text": "请关上窗户",
                            "label_id": "object:original:window",
                            "expected_executable": True,
                            "object_mentions": ["窗户"],
                            "operation_evidence": ["关上"],
                            "semantic_rationale": "关窗",
                        },
                    ],
                    ensure_ascii=False,
                )

        with tempfile.TemporaryDirectory(dir=gen.DEFAULT_OUTPUT_DIR) as temporary:
            output_dir = Path(temporary)
            spec = gen.BatchSpec(
                label="object:original:window",
                kind="object",
                prompt="ONLY THIS LABEL",
                safety_class=None,
            )
            client = FakeClient()
            first = gen.run_batch(
                spec,
                client=client,
                output_dir=output_dir,
                count=2,
                seed=123,
                generation_parameters={"temperature": 0.7},
                model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
                known_labels={"object:original:window"},
            )
            second = gen.run_batch(
                spec,
                client=client,
                output_dir=output_dir,
                count=2,
                seed=123,
                generation_parameters={"temperature": 0.7},
                model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
                known_labels={"object:original:window"},
            )

            self.assertEqual(client.calls, 1)
            self.assertFalse(first["resumed"])
            self.assertTrue(second["resumed"])
            self.assertEqual(first["parsed_count"], 2)
            self.assertEqual(first["rejected_count"], 0)
            self.assertEqual(len(list((output_dir / "raw").glob("*.json"))), 1)
            self.assertEqual(len(list((output_dir / "candidates").glob("*.json"))), 1)

    def test_parser_rejects_a_safety_batch_with_wrong_executable_mix(self) -> None:
        raw = json.dumps(
            [
                {
                    "text": "先别开窗了",
                    "label_id": None,
                    "expected_executable": False,
                    "object_mentions": ["窗"],
                    "operation_evidence": [],
                    "semantic_rationale": "撤回",
                },
                {
                    "text": "这个不用处理",
                    "label_id": None,
                    "expected_executable": False,
                    "object_mentions": [],
                    "operation_evidence": [],
                    "semantic_rationale": "无动作",
                },
            ],
            ensure_ascii=False,
        )

        records = gen.parse_batch_response(
            raw,
            batch_label="safety:multi_clause_turn_correction",
            safety_class="multi_clause_turn_correction",
            count=2,
            seed=1,
            prompt_sha256="p" * 64,
            generation_parameters={},
            model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
            known_labels={"object:original:window"},
        )

        self.assertTrue(all(record["parse_status"] == "rejected" for record in records))
        self.assertTrue(
            all("safety_mix_mismatch" in record["rejection_reasons"] for record in records)
        )

    def test_default_pilot_specs_use_twelve_real_catalog_labels_and_nine_safety_classes(self) -> None:
        self.assertTrue(hasattr(gen, "CORE_LABELS"))
        self.assertTrue(hasattr(gen, "build_batch_specs"))
        object_entry = lambda label, name: {
            "label": label,
            "entry_type": "object",
            "object_name_zh": name,
            "task": "task",
            "detailed_actions": ["操作"],
        }
        scene_entry = lambda label: {
            "label": label,
            "entry_type": "scene",
            "command": "整体目标",
            "objects": [{"object_name_zh": "物体"}],
        }
        entries = []
        for label in gen.CORE_LABELS:
            if label.startswith("scene:"):
                entries.append(scene_entry(label))
            else:
                entries.append(object_entry(label, label.rsplit(":", 1)[-1]))

        specs = gen.build_batch_specs(entries, labels=None, count=12)

        self.assertEqual(len(specs), 21)
        self.assertIn("object:vla:vla_011", [spec.label for spec in specs])
        self.assertNotIn("object:original:stove", [spec.label for spec in specs])
        self.assertEqual(sum(spec.kind == "safety" for spec in specs), 9)

    def test_pilot_runner_writes_manifest_and_rejected_log_with_exact_counts(self) -> None:
        self.assertTrue(hasattr(gen, "run_pilot"))
        gen.DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        entries = [
            {
                "label": "object:original:window",
                "entry_type": "object",
                "object_name_zh": "窗户",
                "task": "window_care",
                "detailed_actions": ["打开窗户"],
            }
        ]

        class FakeClient:
            def generate(self, prompt: str, **kwargs: object) -> str:
                return json.dumps(
                    [
                        {
                            "text": "把窗户打开",
                            "label_id": "object:original:window",
                            "expected_executable": True,
                            "object_mentions": ["窗户"],
                            "operation_evidence": ["打开"],
                            "semantic_rationale": "明确命令",
                        },
                        {
                            "text": "这条缺少操作 span",
                            "label_id": "object:original:window",
                            "expected_executable": True,
                            "object_mentions": ["窗户"],
                            "operation_evidence": [],
                            "semantic_rationale": "故意无效",
                        },
                    ],
                    ensure_ascii=False,
                )

        with tempfile.TemporaryDirectory(dir=gen.DEFAULT_OUTPUT_DIR) as temporary:
            output_dir = Path(temporary)
            manifest = gen.run_pilot(
                entries=entries,
                client=FakeClient(),
                output_dir=output_dir,
                labels=["object:original:window"],
                count=2,
                seed=123,
                generation_parameters={"temperature": 0.7},
                model_revisions={"hf": "h" * 40, "modelscope": "m" * 40},
                input_hashes={"catalog": "c" * 64, "guidelines": "g" * 64},
                model_manifest_sha256="x" * 64,
                script_sha256="s" * 64,
            )

            self.assertEqual(manifest["batch_count"], 1)
            self.assertEqual(manifest["raw_count"], 2)
            self.assertEqual(manifest["parsed_count"], 1)
            self.assertEqual(manifest["rejected_count"], 1)
            self.assertEqual(manifest["requirement_corrections"], gen.REQUIREMENT_CORRECTIONS)
            self.assertTrue((output_dir / "pilot_manifest.json").is_file())
            rejected = json.loads((output_dir / "rejected_candidates.json").read_text("utf-8"))
            self.assertEqual(len(rejected), 1)

    def test_cli_supports_labels_candidate_count_and_dry_run_without_loading_model(self) -> None:
        self.assertTrue(hasattr(gen, "main"))
        loaded = False

        def forbidden_loader(*args: object, **kwargs: object) -> object:
            nonlocal loaded
            loaded = True
            raise AssertionError("dry-run must not load Qwen")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = gen.main(
                [
                    "--labels",
                    "object:original:window",
                    "--candidates-per-label",
                    "2",
                    "--dry-run",
                ],
                model_loader=forbidden_loader,
            )

        self.assertEqual(status, 0)
        self.assertFalse(loaded)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["batch_count"], 1)
        self.assertEqual(plan["candidate_count"], 2)

    def test_module_entrypoint_can_execute_dry_run_after_all_definitions_exist(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            sys,
            "argv",
            ["generate_spoken_annotation_candidates.py", "--dry-run"],
        ), contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                runpy.run_module(
                    "tools.generate_spoken_annotation_candidates",
                    run_name="__main__",
                )

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(json.loads(output.getvalue())["batch_count"], 21)


if __name__ == "__main__":
    unittest.main()
