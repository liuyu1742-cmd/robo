"""Generate acceptance artifacts from the canonical household catalog."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

try:
    from tools.household_catalog import load_household_catalog, validate_catalog
    from tools.build_compliance_catalog import action_plan_for
    from tools.operation_plans import DEFAULT_OPERATION_BY_TASK, canonical_acceptance_action_plan, operation_plan
except ModuleNotFoundError:
    from household_catalog import load_household_catalog, validate_catalog
    from build_compliance_catalog import action_plan_for
    from operation_plans import DEFAULT_OPERATION_BY_TASK, canonical_acceptance_action_plan, operation_plan


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATASETS = ROOT / "datasets"
NEW_TASK_MANIFEST = DATASETS / "new_household_tasks" / "import_manifest.json"

BEDROOM_INSTRUCTIONS = {
    "bed": "铺平床面并整理床铺",
    "pillow": "拍松枕头、摆正朝向并放回床头",
    "quilt": "展开并铺平被子，抚平褶皱",
    "mattress": "检查并拉直床垫边缘，不抓取整张床垫",
    "nightstand": "清空遮挡后摆正床头柜上的物品",
    "bedroom_lamp": "打开卧室灯并确认照明状态",
    "clothes_hanger": "把衣架挂回衣柜横杆",
    "laundry_basket": "把洗衣篮移到指定收纳位置",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def catalog_parts():
    catalog = load_household_catalog()
    errors = validate_catalog(catalog)
    if errors:
        raise ValueError("invalid canonical catalog: " + "; ".join(errors))
    names = {task["id"]: task["name_zh"] for task in catalog["acceptance_tasks"]}
    by_task = defaultdict(list)
    for relation in catalog["task_object_relations"]:
        by_task[relation["task_id"]].append(relation)
    return catalog, names, by_task


def concrete_instruction(task: str, obj: str) -> str:
    if task == "bedroom_service":
        return BEDROOM_INSTRUCTIONS[obj]
    if task == "maintenance_management":
        return f"检查并养护 {obj}"
    if task == "entertainment_service":
        return f"使用 {obj} 提供娱乐服务"
    if task == "cleaning":
        return f"清洁 {obj}"
    return f"执行 {task}：操作 {obj}"


def relation_row(relation: dict, names: dict[str, str]) -> dict:
    acceptance = relation["task_id"]
    task = relation.get("legacy_task_id", acceptance)
    obj = relation["object_id"]
    if task in {"maintenance_management", "entertainment_service"}:
        _, actions = canonical_acceptance_action_plan(task, obj)
    else:
        _, actions = action_plan_for(task, obj)
    return {
        "task": task,
        "acceptance_task": acceptance,
        "task_name_zh": names[acceptance],
        "object": obj,
        "instruction_example": concrete_instruction(acceptance, obj),
        "action_sequence_template": actions,
    }


def build_coverage_payload() -> dict:
    catalog, names, _ = catalog_parts()
    rows = [relation_row(relation, names) for relation in catalog["task_object_relations"]]
    return {
        "format": "canonical_household_operation_coverage_v1",
        "source_catalog": "meta/household_task_catalog.json",
        "acceptance_task_count": len(catalog["acceptance_tasks"]),
        "cleaning_object_count": len({r["object"] for r in rows if r["acceptance_task"] == "cleaning"}),
        "unique_object_count": len(catalog["objects"]),
        "task_object_relation_count": len(rows),
        "items": rows,
    }


def build_coverage(out_json: Path | None = None, out_md: Path | None = None):
    catalog, names, _ = catalog_parts()
    payload = build_coverage_payload()
    rows = payload["items"]
    out_json = out_json or DATASETS / "object_operation_coverage_120.json"
    dump_json(out_json, payload)

    rows_by_task = defaultdict(list)
    for row in rows:
        rows_by_task[row["acceptance_task"]].append(row)
    lines = [
        "# 家庭任务物体操作覆盖表", "",
        "来源：`meta/household_task_catalog.json`（canonical catalog）。", "",
        f"- 验收任务数：{payload['acceptance_task_count']}",
        f"- cleaning 关联物体数：{payload['cleaning_object_count']}",
        f"- 全局唯一物体数：{payload['unique_object_count']}",
        f"- 任务-物体关系数：{payload['task_object_relation_count']}", "",
        "物体在全局目录中只计一次，但同一物体可以关联多个任务，因此关系数可以大于唯一物体数。", "",
    ]
    for acceptance in catalog["acceptance_tasks"]:
        task_id = acceptance["id"]
        lines.extend([f"## {names[task_id]} `{task_id}`", "", "| 序号 | 兼容 task | 物体 | 指令 | 动作序列 |", "|---:|---|---|---|---|"])
        for index, row in enumerate(rows_by_task[task_id], 1):
            actions = " → ".join(row["action_sequence_template"])
            lines.append(f"| {index} | `{row['task']}` | `{row['object']}` | {row['instruction_example']} | `{actions}` |")
        lines.append("")
    out_md = out_md or DOCS / "120_OBJECT_OPERATION_COVERAGE.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def build_demo_cases():
    catalog, names, by_task = catalog_parts()
    cases = []
    for acceptance in catalog["acceptance_tasks"]:
        task_id = acceptance["id"]
        row = relation_row(by_task[task_id][0], names)
        cases.append({
            "id": f"demo_{task_id}_{row['object']}",
            "task": row["task"],
            "acceptance_task": task_id,
            "task_name_zh": names[task_id],
            "source": "canonical_acceptance_template",
            "object": row["object"],
            "target": "none",
            "instruction": row["instruction_example"],
            "predicted_actions": row["action_sequence_template"],
        })
    out_json = DATASETS / "final_demo_cases_15tasks.json"
    dump_json(out_json, cases)
    lines = [
        "# 15 个验收任务最终演示样例", "",
        "每个 acceptance task 至少提供一个代表动作案例。cleaning 样例保留旧清洁子任务 `task`，同时标记 `acceptance_task=cleaning`。", "",
        "| 序号 | 验收任务 | 兼容 task | 指令 | 对象 | 动作序列 |", "|---:|---|---|---|---|---|",
    ]
    for index, case in enumerate(cases, 1):
        actions = " → ".join(case["predicted_actions"])
        lines.append(f"| {index} | `{case['acceptance_task']}` | `{case['task']}` | {case['instruction']} | `{case['object']}` | `{actions}` |")
    out_md = DOCS / "FINAL_DEMO_CASES.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def build_acceptance_instruction_benchmark():
    """Freeze one executable case per canonical task-object relation."""
    payload = build_coverage_payload()
    legacy_benchmark = load_json(DATASETS / "standard_instruction_action_benchmark.json")
    records = []
    for index, row in enumerate(payload["items"], 1):
        acceptance = row["acceptance_task"]
        obj = row["object"]
        if acceptance in {"maintenance_management", "entertainment_service"}:
            instruction = f"执行 {acceptance}：操作 {obj}"
        else:
            matching = next(
                record
                for record in legacy_benchmark
                if record["task"] == row["task"]
                and record["object"] == obj
                and record["actions"] == row["action_sequence_template"]
                and record.get("instruction_style") == "direct"
            )
            instruction = matching["instruction"]
        records.append(
            {
                "id": f"acceptance_{index:03d}_{acceptance}_{obj}",
                "relation_key": f"{acceptance}::{obj}",
                "instruction": instruction,
                "acceptance_task": acceptance,
                "legacy_task_id": row["task"] if acceptance == "cleaning" else None,
                "object": obj,
                "target": "none",
                "actions": row["action_sequence_template"],
            }
        )
    path = DATASETS / "acceptance_instruction_action_benchmark_15tasks.json"
    dump_json(path, records)
    return path


def build_catalog_document():
    catalog, _, by_task = catalog_parts()
    lines = [
        "# 家庭任务 Canonical Catalog（15 个验收任务）", "", "来源：`meta/household_task_catalog.json`。", "",
        f"- 验收任务数：{len(catalog['acceptance_tasks'])}",
        f"- cleaning 关联物体数：{len(by_task['cleaning'])}",
        f"- 全局唯一物体数：{len(catalog['objects'])}",
        f"- 任务-物体关系数：{len(catalog['task_object_relations'])}", "",
        "## 计数语义", "",
        "每个物体 ID 在全局唯一物体表中只计一次；同一物体可关联多个任务。共享物体不会重复增加全局物体数，但会形成多条任务-物体关系。", "",
        "| 验收任务 | 关联物体数 | 物体 |", "|---|---:|---|",
    ]
    for task in catalog["acceptance_tasks"]:
        objects = [relation["object_id"] for relation in by_task[task["id"]]]
        lines.append(f"| `{task['id']}` | {len(objects)} | {', '.join(f'`{obj}`' for obj in objects)} |")
    lines.extend(["", "## 卧室服务的具体操作", ""])
    lines.extend(f"- `{obj}`：{text}" for obj, text in BEDROOM_INSTRUCTIONS.items())
    path = DOCS / "HOUSEHOLD_TASK_CATALOG_15.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_new_task_report():
    manifest = load_json(NEW_TASK_MANIFEST)
    if manifest.get("scope") != "datasets/midterm_15task_delivery":
        raise ValueError("new-task manifest scope must be datasets/midterm_15task_delivery")
    candidates = manifest.get("candidates", [])
    for candidate in candidates:
        for source in candidate.get("source_dataset", []):
            if not source.startswith("datasets/midterm_15task_delivery/"):
                raise ValueError(f"outside midterm-only evidence scope: {source}")
    by_task = defaultdict(list)
    for candidate in candidates:
        by_task[candidate["associated_task_id"]].append(candidate)
    lines = [
        "# 新任务本地数据报告", "",
        "本报告仅使用 `datasets/midterm_15task_delivery` 内的证据，并由最新轻量 manifest `datasets/new_household_tasks/import_manifest.json` 生成；未搜索或汇总其他 datasets。", "",
        f"- maintenance_management 关联物体：{len(by_task['maintenance_management'])}",
        f"- entertainment_service 关联物体：{len(by_task['entertainment_service'])}", "",
        "共享物体沿用 canonical 全局 ID，只计一次全局唯一物体；关联到多个任务时分别计入任务-物体关系。", "",
        "| 新任务 | 物体 | 状态 | 图像 | 视频 | 标注 | 证据路径 |", "|---|---|---|---:|---:|---:|---|",
    ]
    for task_id in ("maintenance_management", "entertainment_service"):
        for item in by_task[task_id]:
            counts = item.get("sample_counts", {})
            sources = "<br>".join(f"`{source}`" for source in item.get("source_dataset", []))
            lines.append(f"| `{task_id}` | `{item['project_object']}` | {item['status']} | {counts.get('images', 0)} | {counts.get('videos', 0)} | {counts.get('annotations', 0)} | {sources} |")
    path = DOCS / "NEW_TASK_LOCAL_DATA_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_diverse_instruction_document():
    coverage = load_json(DATASETS / "object_operation_coverage_120.json")
    lines = [
        "# Canonical 家庭任务多样指令与动作解析", "",
        "本表从 canonical catalog 的任务-物体关系生成。全局唯一物体可服务多个任务，关系行不等同于唯一物体数。", "",
        "## 卧室操作示例", "", *[f"- {text}（`{obj}`）" for obj, text in BEDROOM_INSTRUCTIONS.items()], "",
        "| 序号 | 验收任务 | 兼容 task | 物体 | 直接指令 | 动作序列 |", "|---:|---|---|---|---|---|",
    ]
    for index, row in enumerate(coverage["items"], 1):
        actions = " → ".join(row["action_sequence_template"])
        lines.append(f"| {index} | `{row['acceptance_task']}` | `{row['task']}` | `{row['object']}` | {row['instruction_example']} | `{actions}` |")
    path = DOCS / "DIVERSE_INSTRUCTION_ACTION_PARSE_120.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "coverage": [str(path) for path in build_coverage()],
        "demo_cases": [str(path) for path in build_demo_cases()],
        "acceptance_instruction_benchmark": str(build_acceptance_instruction_benchmark()),
        "catalog_document": str(build_catalog_document()),
        "new_task_report": str(build_new_task_report()),
        "diverse_instruction_document": str(build_diverse_instruction_document()),
    }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
