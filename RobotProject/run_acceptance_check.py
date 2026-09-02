import json
import csv
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "datasets" / "acceptance_check_report.json"
OUT_MD = ROOT / "docs" / "ACCEPTANCE_CHECK_REPORT.md"


REQUIRED_FILES = {
    "正式验收指令基准": ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json",
    "指令动作模型": ROOT / "models" / "instruction_action_transformer_v1.pt",
    "指令动作正式报告": ROOT / "datasets" / "instruction_action_benchmark_report.json",
    "正式动作模型": ROOT / "models" / "action_transformer_project_final.pt",
    "当前视觉模型": ROOT
    / "models"
    / "coco_household"
    / "yolov8n_16cls_small_object_boost_e10"
    / "weights"
    / "best.pt",
    "动作模型评估报告": ROOT / "datasets" / "project_model_final_normalized_eval_report.json",
    "最终指标 JSON": ROOT / "datasets" / "final_metrics_report.json",
    "最终指标 Markdown": ROOT / "datasets" / "final_metrics_report.md",
    "120 物体覆盖 JSON": ROOT / "datasets" / "object_operation_coverage_120.json",
    "120 物体覆盖文档": ROOT / "docs" / "120_OBJECT_OPERATION_COVERAGE.md",
    "15 任务演示 JSON": ROOT / "datasets" / "final_demo_cases_15tasks.json",
    "15 任务演示文档": ROOT / "docs" / "FINAL_DEMO_CASES.md",
    "项目结构说明": ROOT / "docs" / "PROJECT_STRUCTURE.md",
    "总交付说明": ROOT / "PROJECT_DELIVERY_SUMMARY.md",
    "PyCharm 运行指南": ROOT / "PYCHARM_RUN_GUIDE.md",
    "最终演示入口": ROOT / "final_demo.py",
    "视觉错误图库": ROOT / "datasets" / "visual_error_gallery" / "visual_error_gallery.md",
    "动作指标图": ROOT / "datasets" / "visual_results" / "action_metrics.png",
    "视觉总体指标图": ROOT / "datasets" / "visual_results" / "vision_overall_metrics.png",
    "视觉弱类图": ROOT / "datasets" / "visual_results" / "vision_weak_classes.png",
    "家政任务 canonical catalog": ROOT / "meta" / "household_task_catalog.json",
    "期中交付索引": ROOT / "datasets" / "midterm_15task_delivery" / "_delivery_index.csv",
    "验收任务目录说明": ROOT
    / "datasets"
    / "midterm_15task_delivery"
    / "README_ACCEPTANCE_TASKS.md",
}


MIDTERM_DELIVERY = ROOT / "datasets" / "midterm_15task_delivery"
MIDTERM_SCOPE = "datasets/midterm_15task_delivery"
ACCEPTANCE_REFERENCE_COUNTS = {
    "cleaning": 24,
    "maintenance_management": 10,
    "entertainment_service": 8,
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def relation_digest(keys) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def file_status():
    rows = []
    for name, path in REQUIRED_FILES.items():
        exists = path.exists()
        rows.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path.is_file() else None,
            }
        )
    return rows


def metric_status():
    status = {}

    final_report = ROOT / "datasets" / "final_metrics_report.json"
    if final_report.exists():
        report = load_json(final_report)
        action = report.get("project_normalized_test", {})
        vision = report.get("vision_detection", {})
        status["action_sequence_accuracy"] = action.get("sequence_accuracy")
        status["action_step_coverage"] = action.get("step_coverage")
        status["vision_stage_model"] = vision.get("stage_model")
        status["vision_recommended_confidence"] = vision.get("recommended_confidence")
        best = vision.get("threshold_sweep_best_f1") or {}
        status["vision_best_f1"] = best.get("f1")

    coverage = ROOT / "datasets" / "object_operation_coverage_120.json"
    if coverage.exists():
        data = load_json(coverage)
        status["coverage_total"] = data.get("unique_object_count", data.get("total"))
        status["coverage_tasks"] = data.get("acceptance_task_count", data.get("tasks"))
        status["visual_supported_objects"] = sum(
            1 for item in data.get("items", []) if item.get("visual_detection_support")
        )

    demos = ROOT / "datasets" / "final_demo_cases_15tasks.json"
    if demos.exists():
        status["demo_case_count"] = len(load_json(demos))

    gallery = ROOT / "datasets" / "visual_error_gallery" / "images"
    status["visual_error_images"] = (
        len(list(gallery.glob("*.jpg"))) if gallery.exists() else 0
    )

    instruction_report = ROOT / "datasets" / "instruction_action_benchmark_report.json"
    if instruction_report.exists():
        data = load_json(instruction_report)
        instruction_metrics = data.get("metrics", {})
        audit = data.get("audit", {})
        status["instruction_exact_sequence_accuracy"] = instruction_metrics.get(
            "exact_sequence_accuracy"
        )
        status["instruction_step_accuracy"] = instruction_metrics.get("step_accuracy")
        status["instruction_parse_accuracy"] = instruction_metrics.get(
            "instruction_parse_accuracy"
        )
        status["instruction_benchmark_cases"] = instruction_metrics.get("cases")
        status["instruction_benchmark_tasks"] = audit.get("tasks")
        status["instruction_benchmark_objects"] = audit.get("objects")
        status["instruction_template_overlap_count"] = audit.get(
            "template_overlap_count"
        )
        status["instruction_benchmark_task_ids"] = sorted(audit.get("task_ids", []))
        report_relation_keys = sorted(
            f"{case['resolved']['acceptance_task']}::{case['resolved']['object']}"
            for case in data.get("cases", [])
            if case.get("resolved")
        )
        status["instruction_report_relation_keys"] = report_relation_keys
        status["instruction_report_relation_digest"] = audit.get("relation_digest")

    instruction_benchmark = ROOT / "datasets" / "acceptance_instruction_action_benchmark_15tasks.json"
    if instruction_benchmark.exists():
        benchmark = load_json(instruction_benchmark)
        benchmark_relation_keys = sorted(
            row.get("relation_key")
            or f"{row['acceptance_task']}::{row['object']}"
            for row in benchmark
        )
        status["instruction_benchmark_relation_keys"] = benchmark_relation_keys
        status["instruction_benchmark_relation_digest"] = relation_digest(
            benchmark_relation_keys
        )

    catalog_path = ROOT / "meta" / "household_task_catalog.json"
    if catalog_path.exists():
        catalog = load_json(catalog_path)
        relations = catalog.get("task_object_relations", [])
        relation_counts = Counter(row.get("task_id") for row in relations)
        compatibility = catalog.get("compatibility_task_map", {})
        canonical_relation_keys = sorted(
            f"{row['task_id']}::{row['object_id']}" for row in relations
        )
        status.update(
            {
                "canonical_acceptance_task_ids": sorted(
                    task["id"] for task in catalog.get("acceptance_tasks", [])
                ),
                "canonical_relation_keys": canonical_relation_keys,
                "canonical_relation_digest": relation_digest(canonical_relation_keys),
                "acceptance_task_count": len(catalog.get("acceptance_tasks", [])),
                "cleaning_object_count": len(
                    {
                        row.get("object_id")
                        for row in relations
                        if row.get("task_id") == "cleaning"
                    }
                ),
                # Count canonical IDs once. Shared task relations must not inflate this.
                "unique_object_count": len(set(catalog.get("objects", []))),
                "task_object_relation_count": len(relations),
                "legacy_cleaning_map_ok": all(
                    compatibility.get(task_id) == "cleaning"
                    for task_id in (
                        "floor_cleaning",
                        "dish_washing",
                        "bathroom_cleaning",
                    )
                ),
                "maintenance_object_count": relation_counts["maintenance_management"],
                "entertainment_object_count": relation_counts["entertainment_service"],
            }
        )

    delivery_index = MIDTERM_DELIVERY / "_delivery_index.csv"
    if delivery_index.exists():
        with delivery_index.open("r", encoding="utf-8-sig", newline="") as handle:
            delivery_rows = list(csv.DictReader(handle))
        reference_rows = [
            row for row in delivery_rows if row.get("is_reference", "").lower() == "true"
        ]
        source_rows = [
            row for row in delivery_rows if row.get("is_reference", "").lower() != "true"
        ]
        count_fields = ("images", "videos", "annotations", "annotation_template_rows")

        def total(rows, field):
            return sum(int(row.get(field) or 0) for row in rows)

        reference_totals = {field: total(reference_rows, field) for field in count_fields}
        all_totals = {field: total(delivery_rows, field) for field in count_fields}
        source_totals = {field: total(source_rows, field) for field in count_fields}
        outside_midterm = sum(
            1
            for row in reference_rows
            if not row.get("source_object_dir", "").replace("\\", "/").startswith(
                MIDTERM_SCOPE + "/"
            )
        )
        valid_directories = 0
        for task_id, expected_count in ACCEPTANCE_REFERENCE_COUNTS.items():
            matches = list(MIDTERM_DELIVERY.glob(f"{task_id}__*"))
            if len(matches) != 1:
                continue
            object_root = matches[0] / "objects"
            object_dirs = list(object_root.iterdir()) if object_root.is_dir() else []
            if len(object_dirs) == expected_count and all(
                (object_dir / "metadata" / "reference.json").is_file()
                or any((object_dir / "images").glob("*"))
                or any(object_dir.glob("*/images/*"))
                for object_dir in object_dirs
            ):
                valid_directories += 1
        status.update(
            {
                "acceptance_reference_directory_count": valid_directories,
                "acceptance_reference_rows": len(reference_rows),
                "reference_image_count": reference_totals["images"],
                "reference_video_count": reference_totals["videos"],
                "reference_annotation_count": reference_totals["annotations"],
                "reference_annotation_template_rows": reference_totals[
                    "annotation_template_rows"
                ],
                "reference_totals_match_source_totals": all_totals == source_totals,
                "local_data_source": MIDTERM_SCOPE,
                "reference_media_copied": sum(reference_totals.values()),
                "reference_paths_outside_midterm": outside_midterm,
                "split_leak_check_applicable": False,
                "split_leak_check_reason": "Reference-only views contain no train/val/test split metadata.",
                "local_unique_image_total": source_totals["images"],
                "local_unique_video_total": source_totals["videos"],
                "local_unique_annotation_total": source_totals["annotations"],
            }
        )

    # Closest existing metric to video-only end-to-end action parsing.
    # Keep it separate from teacher-forced sequence accuracy, which receives
    # the correct task/object conditions.
    visual_conditions = ROOT / "datasets" / "epic_visual_condition_report.json"
    if visual_conditions.exists():
        data = load_json(visual_conditions)
        best = data.get("best_validation_metrics", {})
        status["video_verb_accuracy"] = best.get("verb_accuracy")
        status["video_task_accuracy"] = best.get("task_accuracy")
        status["video_object_accuracy"] = best.get("object_accuracy")
        status["video_joint_accuracy"] = best.get("joint_accuracy")
    return status


def judge(files, metrics):
    benchmark_task_ids = set(metrics.get("instruction_benchmark_task_ids") or [])
    canonical_task_ids = set(metrics.get("canonical_acceptance_task_ids") or [])
    missing_task_ids = sorted(canonical_task_ids - benchmark_task_ids)
    unexpected_task_ids = sorted(benchmark_task_ids - canonical_task_ids)
    canonical_relation_keys = set(metrics.get("canonical_relation_keys") or [])
    benchmark_relation_keys = set(
        metrics.get("instruction_benchmark_relation_keys") or []
    )
    report_relation_keys = set(metrics.get("instruction_report_relation_keys") or [])
    missing_relation_keys = sorted(canonical_relation_keys - benchmark_relation_keys)
    unexpected_relation_keys = sorted(benchmark_relation_keys - canonical_relation_keys)
    report_missing_relation_keys = sorted(canonical_relation_keys - report_relation_keys)
    report_unexpected_relation_keys = sorted(report_relation_keys - canonical_relation_keys)
    relation_digests_match = (
        bool(metrics.get("canonical_relation_digest"))
        and metrics.get("instruction_benchmark_relation_digest")
        == metrics.get("canonical_relation_digest")
        and metrics.get("instruction_report_relation_digest")
        == metrics.get("canonical_relation_digest")
    )
    missing = [row for row in files if not row["exists"]]
    action_ok = (metrics.get("instruction_exact_sequence_accuracy") or 0) >= 0.9
    demo_ok = metrics.get("demo_case_count") == 15
    gallery_ok = (metrics.get("visual_error_images") or 0) > 0
    task_hierarchy_ok = (
        metrics.get("acceptance_task_count") == 15
        and metrics.get("cleaning_object_count") == 24
        and metrics.get("legacy_cleaning_map_ok") is True
        and metrics.get("maintenance_object_count") == 10
        and metrics.get("entertainment_object_count") == 8
    )
    unique_object_scope_ok = (metrics.get("unique_object_count") or 0) >= 124
    relation_count_ok = metrics.get("task_object_relation_count") == 139
    local_data_integrity_ok = (
        metrics.get("acceptance_reference_directory_count") == 3
        and metrics.get("acceptance_reference_rows") == 38
        and metrics.get("reference_image_count") == 0
        and metrics.get("reference_video_count") == 0
        and metrics.get("reference_annotation_count") == 0
        and metrics.get("reference_annotation_template_rows") == 0
        and metrics.get("reference_totals_match_source_totals") is True
        and metrics.get("local_data_source") == MIDTERM_SCOPE
        and metrics.get("reference_media_copied") == 0
        and metrics.get("reference_paths_outside_midterm") == 0
        and metrics.get("split_leak_check_applicable") is False
        and bool(metrics.get("split_leak_check_reason"))
    )
    instruction_benchmark_ok = (
        action_ok
        and metrics.get("instruction_benchmark_tasks") == 15
        and metrics.get("instruction_benchmark_objects") == 124
        and metrics.get("instruction_benchmark_cases") == 139
        and metrics.get("instruction_template_overlap_count") == 0
        and not missing_task_ids
        and not unexpected_task_ids
        and len(canonical_relation_keys) == 139
        and len(benchmark_relation_keys) == 139
        and len(report_relation_keys) == 139
        and not missing_relation_keys
        and not unexpected_relation_keys
        and not report_missing_relation_keys
        and not report_unexpected_relation_keys
        and relation_digests_match
    )
    artifact_ok = not missing and demo_ok and gallery_ok
    video_joint_ok = (metrics.get("video_joint_accuracy") or 0) >= 0.9
    return {
        # The confirmed task scope scores standard instruction -> complete
        # action sequence. Video-only joint recognition remains informational.
        "passed": (
            artifact_ok
            and task_hierarchy_ok
            and unique_object_scope_ok
            and relation_count_ok
            and local_data_integrity_ok
            and instruction_benchmark_ok
        ),
        "artifact_and_template_check_passed": artifact_ok,
        "task_hierarchy_passed": task_hierarchy_ok,
        "unique_object_scope_passed": unique_object_scope_ok,
        "task_object_relation_check_passed": relation_count_ok,
        "local_data_integrity_passed": local_data_integrity_ok,
        "instruction_benchmark_check_passed": instruction_benchmark_ok,
        "instruction_benchmark_missing_task_ids": missing_task_ids,
        "instruction_benchmark_unexpected_task_ids": unexpected_task_ids,
        "instruction_benchmark_missing_relation_keys": missing_relation_keys,
        "instruction_benchmark_unexpected_relation_keys": unexpected_relation_keys,
        "instruction_report_missing_relation_keys": report_missing_relation_keys,
        "instruction_report_unexpected_relation_keys": report_unexpected_relation_keys,
        "instruction_relation_digests_match": relation_digests_match,
        "video_joint_90_informational": video_joint_ok,
        "missing_count": len(missing),
        "action_accuracy_ok": action_ok,
        "task_object_relation_count": metrics.get("task_object_relation_count"),
        "demo_ok": demo_ok,
        "visual_gallery_ok": gallery_ok,
    }


def pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def main():
    files = file_status()
    metrics = metric_status()
    result = judge(files, metrics)

    report = {
        "project_root": str(ROOT),
        "acceptance_status": result,
        "metrics": metrics,
        "required_files": files,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 一键验收检查报告",
        "",
        f"- 总体状态：{'通过' if result['passed'] else '未通过'}",
        f"- 缺失文件数：{result['missing_count']}",
        "",
        "## 核心指标",
        "",
        f"- 冻结指令整序列准确率：{pct(metrics.get('instruction_exact_sequence_accuracy'))}",
        f"- 冻结指令步骤准确率：{pct(metrics.get('instruction_step_accuracy'))}",
        f"- 指令字段解析准确率：{pct(metrics.get('instruction_parse_accuracy'))}",
        f"- 冻结测试样例数：{metrics.get('instruction_benchmark_cases', 'N/A')}",
        f"- 冻结基准任务/物体覆盖：{metrics.get('instruction_benchmark_tasks', 'N/A')}/{metrics.get('instruction_benchmark_objects', 'N/A')}",
        f"- 指令模板跨集合重叠数：{metrics.get('instruction_template_overlap_count', 'N/A')}",
        f"- 指定90%要求：{'通过' if result['instruction_benchmark_check_passed'] else '未通过'}",
        f"- 验收任务数：{metrics.get('acceptance_task_count', 'N/A')}（要求恰好 15）",
        f"- cleaning 关联物体数：{metrics.get('cleaning_object_count', 'N/A')}（要求 24）",
        f"- 全局唯一物体数：{metrics.get('unique_object_count', 'N/A')}（要求至少 124）",
        f"- 任务-物体关系数：{metrics.get('task_object_relation_count', 'N/A')}（当前要求 139）",
        f"- maintenance/entertainment 关联数：{metrics.get('maintenance_object_count', 'N/A')}/{metrics.get('entertainment_object_count', 'N/A')}（要求 8/6）",
        f"- 旧 cleaning ID 映射：{'通过' if metrics.get('legacy_cleaning_map_ok') else '未通过'}",
        f"- 轻量引用目录/行：{metrics.get('acceptance_reference_directory_count', 'N/A')}/{metrics.get('acceptance_reference_rows', 'N/A')}（要求 3/38）",
        f"- 引用行图片/视频/标注数：{metrics.get('reference_image_count', 'N/A')}/{metrics.get('reference_video_count', 'N/A')}/{metrics.get('reference_annotation_count', 'N/A')}",
        f"- 本地唯一图片/视频/标注总数：{metrics.get('local_unique_image_total', 'N/A')}/{metrics.get('local_unique_video_total', 'N/A')}/{metrics.get('local_unique_annotation_total', 'N/A')}",
        f"- 本地数据来源：{metrics.get('local_data_source', 'N/A')}",
        f"- 引用媒体复制数：{metrics.get('reference_media_copied', 'N/A')}",
        f"- 期中目录外引用数：{metrics.get('reference_paths_outside_midterm', 'N/A')}",
        f"- split 泄漏检查适用：{metrics.get('split_leak_check_applicable', 'N/A')}（{metrics.get('split_leak_check_reason', 'N/A')}）",
        f"- 任务层级检查：{'通过' if result['task_hierarchy_passed'] else '未通过'}",
        f"- 唯一物体范围检查：{'通过' if result['unique_object_scope_passed'] else '未通过'}",
        f"- 本地数据完整性检查：{'通过' if result['local_data_integrity_passed'] else '未通过'}",
        f"- 旧教师强制动作序列准确率（参考）：{pct(metrics.get('action_sequence_accuracy'))}",
        f"- 旧动作步骤覆盖率（参考）：{pct(metrics.get('action_step_coverage'))}",
        f"- 仅视频动词准确率：{pct(metrics.get('video_verb_accuracy'))}",
        f"- 仅视频任务准确率：{pct(metrics.get('video_task_accuracy'))}",
        f"- 仅视频物体准确率：{pct(metrics.get('video_object_accuracy'))}",
        f"- 仅视频联合准确率：{pct(metrics.get('video_joint_accuracy'))}",
        f"- 视频联合90%（扩展研究，不作为本轮门槛）：{'达到' if result['video_joint_90_informational'] else '未达到'}",
        f"- 120 物体覆盖：{metrics.get('coverage_total', 'N/A')}",
        f"- 15 类任务演示样例：{metrics.get('demo_case_count', 'N/A')}",
        f"- 视觉直接检测覆盖对象数：{metrics.get('visual_supported_objects', 'N/A')}",
        f"- 视觉最优 F1：{pct(metrics.get('vision_best_f1'))}",
        f"- 推荐视觉阈值：conf={metrics.get('vision_recommended_confidence', 'N/A')}",
        f"- 视觉错误图库图片数：{metrics.get('visual_error_images', 0)}",
        "",
        "## 文件检查",
        "",
        "| 项目 | 状态 | 路径 |",
        "|---|---|---|",
    ]
    for row in files:
        lines.append(
            f"| {row['name']} | {'存在' if row['exists'] else '缺失'} | `{row['path']}` |"
        )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
