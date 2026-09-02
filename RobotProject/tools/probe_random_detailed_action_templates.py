"""Developer probe for runtime-only grammar acceptance (never reads data splits)."""

from tools.detailed_action_catalog import load_detailed_action_catalog
from tools.detailed_action_random_cases import _TASK_VERBS
from tools.detailed_action_runtime import DetailedActionParser


def main() -> None:
    parser = DetailedActionParser(semantic_backend="bge", device="cuda")
    entries = load_detailed_action_catalog()["entries"]
    objects = [entry for entry in entries if entry["entry_type"] == "object"]
    scenes = [entry for entry in entries if entry["entry_type"] == "scene"]
    object_forms = (
        lambda e: f"请{_TASK_VERBS.get(e['task'], ('处理', '处理'))[0]}{e['object_name_zh']}",
        lambda e: f"把{e['object_name_zh']}{_TASK_VERBS.get(e['task'], ('处理', '处理'))[1]}",
        lambda e: f"请对{e['object_name_zh']}执行{e['task']}操作",
        lambda e: f"请按细分步骤处理{e['object_name_zh']}",
    )
    scene_forms = (
        lambda e: e["command"] + "！",
        lambda e: e["command"].replace("，", "、") + "。",
        lambda e: "请执行" + e["command"],
        lambda e: e["command"] + "，请联动执行",
    )
    for index, form in enumerate(object_forms):
        results = [parser.resolve(form(entry)) for entry in objects]
        print("OBJECT", index, sum(row.get("label") == entry["label"] for row, entry in zip(results, objects)), sum(row.get("result_type") != "clarification_required" for row in results))
    for index, form in enumerate(scene_forms):
        results = [parser.resolve(form(entry)) for entry in scenes]
        print("SCENE", index, sum(row.get("label") == entry["label"] for row, entry in zip(results, scenes)), sum(row.get("result_type") != "clarification_required" for row in results))
        if index == 0:
            for row, entry in zip(results, scenes):
                if row.get("label") != entry["label"]:
                    print("SCENE_MISS", entry["label"], entry["command"], row.get("reason"), row.get("label"))


if __name__ == "__main__":
    main()
