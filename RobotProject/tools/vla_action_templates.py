"""Read-only VLA workbook action templates for instruction-result presentation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


TASKS = (
    ("indoor_cleaning", "室内卫生清洁"),
    ("cooking_heating", "烹饪与加热辅助"),
    ("item_delivery", "物品递送"),
    ("organizing_storage", "整理收纳"),
    ("storage_open_close", "储物设施开合"),
    ("indoor_installation", "室内安装与布置"),
    ("tableware_placement", "餐具与容器摆放"),
    ("clothing_entryway", "衣物鞋类与玄关归位"),
    ("workspace_service", "工作学习区服务"),
    ("bathroom_personal_care", "卫浴用品与个人卫生服务"),
)

# Natural equivalents derived from the same reviewed workbook operation labels.
# They only relax wording (``放入`` -> ``放到`` and an optional room qualifier);
# the selected task, object, and action template remain workbook-owned.
SOURCE_OPERATION_ALIASES = {
    ("牙膏", "放入洗手台上的杯中"): ("放到洗手台的杯中",),
    ("数码相机", "数码相机安装到卧室三脚架"): ("安装到三脚架",),
}

# Distinctive phrases used by the colloquial test generator.  Some workbook
# operations mention several catalogued objects in one sentence (for example a
# knife, chopping board and bowl).  Object-name matching alone cannot tell
# which workbook row the command is exercising, so these phrases provide an
# explicit, human-readable tie-breaker without exposing task IDs to the user.
NATURAL_MATCH_ALIASES = {
    "vla_001": ("把厨房垃圾桶放稳",),
    "vla_005": ("客厅里的几罐饮料",),
    "vla_006": ("用削皮刀把洋葱",),
    "vla_012": ("做完饭把烤箱门关好",),
    "vla_016": ("在砧板上把洋葱",),
    "vla_017": ("平底锅里的苹果盛到碗里",),
    "vla_022": ("用餐刀把洋葱",),
    "vla_023": ("床上的书放到床头柜",),
    "vla_027": ("从厨房冰箱拿两瓶饮料",),
    "vla_030": ("床上的书收到床头柜",),
    "vla_031": ("盒装饮料收进橱柜里的收纳盒",),
    "vla_032": ("六本书收到客厅地面的收纳箱",),
    "vla_033": ("卧室收纳篮里的物品分类",),
    "vla_034": ("六本书收到客厅地面的纸箱",),
    "vla_043": ("烤箱不用了，把烤箱门关好",),
    "vla_045": ("海报挂到厨房的墙钉",),
    "vla_046": ("数码相机安装到卧室的三脚架",),
    "vla_047": ("厨房台面上的海报挂到墙钉",),
    "vla_048": ("用相机三脚架接住",),
    "vla_058": ("平底锅里的苹果盛到盘子里",),
    "vla_060": ("台面上的两顶棒球帽",),
    "vla_062": ("启动洗衣机",),
    "vla_063": ("走廊地面的两双运动鞋",),
    "vla_064": ("把鞋架整理好",),
    "vla_076": ("把卫生巾盒放到卫浴搁板",),
    "vla_077": ("卧室收纳篮里的卫生巾盒",),
}

VLA_NATURAL_GOAL_OVERRIDES = {
    "vla_001": "把厨房垃圾桶放稳，再把客厅里的汽水罐扔进去",
    "vla_005": "把客厅里的几罐饮料拿到厨房，扔进垃圾桶",
    "vla_006": "用削皮刀把洋葱切成丁，切好放进碗里，刀和砧板用完放回水槽",
    "vla_007": "把卷心菜和辣椒切成丁，再放进炉灶上的平底锅里烹调",
    "vla_010": "把蔬菜按类别分别放进三个混合碗",
    "vla_011": "把机器人移动到厨房炉灶旁",
    "vla_012": "做完饭把烤箱门关好",
    "vla_013": "把烤面包机的下层烤盘完全推回去",
    "vla_014": "把烤箱上层的烹饪托盘完全拉出来",
    "vla_015": "打开橱柜和玻璃罐，把熟香肠放进罐里，再关好罐子放回橱柜",
    "vla_016": "在砧板上把洋葱切成丁，切好放进碗里，刀和砧板用完放回水槽",
    "vla_017": "把平底锅里的苹果盛到碗里",
    "vla_022": "用餐刀把洋葱切成丁，切好放进碗里，刀和砧板用完放回水槽",
    "vla_023": "把床上的书放到床头柜，再把两只凉鞋并排放在床边",
    "vla_027": "从厨房冰箱拿两瓶饮料放到客厅咖啡台，再把冰箱门关好",
    "vla_030": "把床上的书收到床头柜，再把凉鞋并排放在床边",
    "vla_031": "把台面上的盒装饮料收进橱柜里的收纳盒",
    "vla_032": "把书架上的六本书收到客厅地面的收纳箱里",
    "vla_033": "把卧室收纳篮里的物品分类放到浴室指定位置",
    "vla_034": "把书架上的六本书收到客厅地面的纸箱里",
    "vla_035": "把冰箱抽屉里的苹果放到冰箱层架上",
    "vla_043": "烤箱不用了，把烤箱门关好",
    "vla_045": "把海报挂到厨房的墙钉上",
    "vla_046": "把数码相机安装到卧室的三脚架上",
    "vla_047": "把厨房台面上的海报挂到墙钉上",
    "vla_048": "用相机三脚架接住并支撑好数码相机",
    "vla_058": "把平底锅里的苹果盛到盘子里",
    "vla_060": "把台面上的两顶棒球帽拿到洗衣机里清洗",
    "vla_061": "把毛巾分别放进两个篮子里",
    "vla_062": "启动洗衣机，把两顶棒球帽洗干净",
    "vla_063": "把走廊地面的两双运动鞋和两双凉鞋拿到鞋架，分类并排放好",
    "vla_064": "把鞋架整理好，运动鞋和凉鞋分开并排放置",
    "vla_069": "把铅笔和两支笔放进笔盒，笔盒留在桌面上",
    "vla_071": "把床上的笔记本电脑拿到书桌上并合好屏幕",
    "vla_076": "把卫生巾盒放到卫浴搁板上，并和其他浴室用品分区摆放",
    "vla_077": "把卧室收纳篮里的卫生巾盒拿到浴室搁板上",
    "vla_079": "把卧室收纳篮里的洗涤剂拿到洗手池下方并排放好",
}


def _source_phrase(value: str) -> str:
    value = value.strip()
    suffixes = {
        "台面": "台面上", "橱柜": "橱柜里", "床": "床上",
        "床上": "床上", "书柜": "书柜里", "书架": "书架上",
        "转椅": "转椅上", "冰箱抽屉": "冰箱抽屉里",
        "厨房冰箱": "厨房冰箱里", "走廊地面": "走廊地面上",
    }
    return suffixes.get(value, value)


def _destination_phrase(value: str) -> str:
    value = value.strip()
    suffixes = {
        "台面": "台面上", "橱柜": "橱柜里", "抽屉": "抽屉里",
        "书桌": "书桌上", "桌面": "桌面上", "笔盒": "笔盒里",
    }
    return suffixes.get(value, value)


def naturalize_vla_goal(relation: Mapping[str, object]) -> str:
    object_id = str(relation.get("object_id") or relation.get("object") or "")
    if not object_id:
        raise KeyError("object_id")
    object_name = str(relation.get("object_name_zh") or relation.get("object_name") or "")
    label = str(relation["source_operation_label"]).strip()
    if object_id in VLA_NATURAL_GOAL_OVERRIDES:
        return VLA_NATURAL_GOAL_OVERRIDES[object_id]

    normalized = label.replace("单件归位", "")
    subject = object_name
    path = normalized
    if "：" in normalized:
        prefix, path = normalized.split("：", 1)
        if object_name in prefix:
            subject = prefix
    if "→" in path:
        source, destination = path.split("→", 1)
        source = _source_phrase(source)
        destination = _destination_phrase(destination)
        if any(term in destination for term in ("橱柜", "抽屉", "盒", "箱")):
            return f"把{source}的{subject}收进{destination}"
        return f"把{source}的{subject}拿到{destination}"

    close_match = re.fullmatch(r"(?:完全)?关闭(.+)", label)
    if close_match:
        return f"把{close_match.group(1)}关好"
    open_match = re.fullmatch(r"打开(.+)", label)
    if open_match:
        return f"把{open_match.group(1)}打开"
    push_match = re.fullmatch(r"将(.+)完全推入", label)
    if push_match:
        return f"把{push_match.group(1)}完全推回去"
    pull_match = re.fullmatch(r"将(.+)完全拉出", label)
    if pull_match:
        return f"把{pull_match.group(1)}完全拉出来"
    if label.startswith(("放在", "放入", "移至", "叠放在")):
        return f"把{object_name}{label}"
    if label.startswith("接收"):
        return f"用{object_name}{label}"
    return f"把{object_name}{label}"

# Workbook rows: sheet | task name | object name | operation label | source row.
# The workbook summary declares ten task groups; the six maintenance-tool rows
# are outside that stated 82-object scope and are intentionally not included.
RAW_ROWS = """
表5-8-1|室内卫生清洁|垃圾桶|多罐汽水投放：客厅→厨房垃圾桶|5
表5-8-1|室内卫生清洁|海绵|海绵取放：台面→橱柜|6
表5-8-1|室内卫生清洁|清洁刷|洗碗刷取放：橱柜→台面|7
表5-8-1|室内卫生清洁|清洁喷雾瓶|喷雾瓶取放：橱柜→台面|8
表5-8-1|室内卫生清洁|饮料罐|多罐汽水投放：客厅→厨房垃圾桶|9
表5-8-2|烹饪与加热辅助|削皮刀|洋葱切丁：水槽→砧板→碗，刀具与砧板回放水槽|5
表5-8-2|烹饪与加热辅助|平底锅|卷心菜和辣椒切丁并在炉灶平底锅中烹调|6
表5-8-2|烹饪与加热辅助|打蛋器|台面→抽屉单件归位|7
表5-8-2|烹饪与加热辅助|擀面杖|台面→抽屉单件归位|8
表5-8-2|烹饪与加热辅助|混合碗|按蔬菜类别分入三个混合碗|9
表5-8-2|烹饪与加热辅助|炉灶|导航至厨房炉灶|10
表5-8-2|烹饪与加热辅助|烤箱|关闭烤箱门|11
表5-8-2|烹饪与加热辅助|烤面包机下层烤盘|将烤箱下层烤盘完全推入|12
表5-8-2|烹饪与加热辅助|烹饪托盘|将烤箱上层托盘完全拉出|13
表5-8-2|烹饪与加热辅助|玻璃罐|开柜、开罐、放入熟香肠、关罐并归回|14
表5-8-2|烹饪与加热辅助|砧板|洋葱切丁：水槽→砧板→碗，刀具与砧板回放水槽|15
表5-8-2|烹饪与加热辅助|碗|苹果：平底锅→碗|16
表5-8-2|烹饪与加热辅助|茶壶|台面→橱柜单件归位|17
表5-8-2|烹饪与加热辅助|调味罐|台面→橱柜单件归位|18
表5-8-2|烹饪与加热辅助|量杯|台面→抽屉单件归位|19
表5-8-2|烹饪与加热辅助|锅|台面→橱柜单件归位|20
表5-8-2|烹饪与加热辅助|餐刀|洋葱切丁：水槽→砧板→碗，刀具与砧板回放水槽|21
表5-8-4|物品递送|书籍|书籍：床上→床头柜；两只凉鞋并排放在床边|5
表5-8-4|物品递送|水瓶|水瓶：橱柜→台面|6
表5-8-4|物品递送|盒装饮料|盒装饮料：橱柜→台面|7
表5-8-4|物品递送|罐装物|罐装物：橱柜→台面|8
表5-8-4|物品递送|饮料瓶|两瓶饮料：厨房冰箱→客厅咖啡台，完成后关冰箱|9
表5-8-4|物品递送|马克杯|马克杯：橱柜→台面|10
表5-8-5|整理收纳|密封保鲜盒|密封保鲜盒：台面→橱柜|5
表5-8-5|整理收纳|床头柜|书籍：床→床头柜，凉鞋并排放置|6
表5-8-5|整理收纳|收纳盒|盒装饮料：台面→橱柜|7
表5-8-5|整理收纳|收纳箱|六本书：书架→客厅地面收纳箱|8
表5-8-5|整理收纳|收纳篮|卧室篮中物品分类归位至浴室指定区域|9
表5-8-5|整理收纳|纸箱|六本书：书架→客厅地面纸箱|10
表5-8-6|储物设施开合|冰箱层架|苹果：冰箱抽屉→冰箱层架|5
表5-8-6|储物设施开合|冰箱抽屉|完全关闭冰箱抽屉|6
表5-8-6|储物设施开合|冰箱门|关闭冰箱门|7
表5-8-6|储物设施开合|微波炉门|关闭微波炉门|8
表5-8-6|储物设施开合|抽屉|关闭左侧抽屉|9
表5-8-6|储物设施开合|橱柜门|关闭橱柜门|10
表5-8-6|储物设施开合|洗碗机上层架|将洗碗机上层架完全推入|11
表5-8-6|储物设施开合|洗碗机门|关闭洗碗机门|12
表5-8-6|储物设施开合|烤箱门|关闭烤箱门|13
表5-8-6|储物设施开合|烤面包机烤箱门|打开烤面包机烤箱门|14
表5-8-7|室内安装与布置|墙钉|海报悬挂到厨房墙钉|5
表5-8-7|室内安装与布置|数码相机|数码相机安装到卧室三脚架|6
表5-8-7|室内安装与布置|海报|厨房台面海报→墙钉悬挂|7
表5-8-7|室内安装与布置|相机三脚架|接收并支撑数码相机|8
表5-8-7|室内安装与布置|蜡烛|蜡烛：橱柜→台面|9
表5-8-8|餐具与容器摆放|咖啡杯|橱柜→台面|5
表5-8-8|餐具与容器摆放|夹子|橱柜→台面|6
表5-8-8|餐具与容器摆放|披萨刀|橱柜→台面|7
表5-8-8|餐具与容器摆放|木勺|台面→橱柜|8
表5-8-8|餐具与容器摆放|水罐|台面→橱柜|9
表5-8-8|餐具与容器摆放|汤勺|台面→橱柜|10
表5-8-8|餐具与容器摆放|洗碗刷|台面→橱柜|11
表5-8-8|餐具与容器摆放|滤盆|台面→橱柜|12
表5-8-8|餐具与容器摆放|盘子|苹果：平底锅→盘子|13
表5-8-8|餐具与容器摆放|铝箔纸|台面→橱柜|14
表5-8-9|衣物鞋类与玄关归位|棒球帽|两顶棒球帽：台面→洗衣机清洗|5
表5-8-9|衣物鞋类与玄关归位|毛巾|毛巾分入两个篮子|6
表5-8-9|衣物鞋类与玄关归位|洗衣机|运行洗衣机清洗两顶棒球帽|7
表5-8-9|衣物鞋类与玄关归位|鞋子|两双运动鞋和两双凉鞋：走廊地面→鞋架并按类别并排|8
表5-8-9|衣物鞋类与玄关归位|鞋架|接收鞋类并保持运动鞋、凉鞋分别并排|9
表5-8-10|工作学习区服务|办公椅|移至书桌旁|5
表5-8-10|工作学习区服务|文件夹|转椅→书桌并置于鼠标旁|6
表5-8-10|工作学习区服务|显示器|放在书桌上|7
表5-8-10|工作学习区服务|电脑|放在书桌下|8
表5-8-10|工作学习区服务|笔盒|铅笔和两支笔→笔盒，笔盒留在桌面|9
表5-8-10|工作学习区服务|笔记本|叠放在文件夹上|10
表5-8-10|工作学习区服务|笔记本电脑|床→书桌并合上电脑|11
表5-8-10|工作学习区服务|订书机|书柜→桌面|12
表5-8-10|工作学习区服务|铅笔|放入笔盒|13
表5-8-10|工作学习区服务|键盘|放在显示器旁|14
表5-8-10|工作学习区服务|鼠标|放在键盘旁|15
表5-8-11|卫浴用品与个人卫生服务|卫浴搁板|接收卫生巾盒并保持浴室用品分区|5
表5-8-11|卫浴用品与个人卫生服务|卫生巾盒|卧室篮→浴室搁板|6
表5-8-11|卫浴用品与个人卫生服务|固体香皂|台面→橱柜|7
表5-8-11|卫浴用品与个人卫生服务|洗涤剂|卧室篮→洗手池下方并排放置|8
表5-8-11|卫浴用品与个人卫生服务|牙刷|放入洗手台上的杯中|9
表5-8-11|卫浴用品与个人卫生服务|牙膏|放入洗手台上的杯中|10
表5-8-11|卫浴用品与个人卫生服务|皂液器|橱柜→台面|11
""".strip()


@dataclass(frozen=True)
class VlaActionCandidate:
    task_id: str
    object_id: str
    object_name_zh: str
    source_operation_label: str
    match_keywords: tuple[str, ...]
    steps: tuple[str, ...]
    source_sheet: str
    source_row: int


def _steps(object_name: str, label: str) -> tuple[str, ...]:
    if "汽水投放" in label:
        return ("拿起汽水罐", "移动至厨房垃圾桶", "投放汽水罐", "确认已放入")
    if "关闭" in label:
        return (f"定位{object_name}", f"执行关闭{object_name}", f"确认{object_name}已关闭")
    if "打开" in label:
        return (f"定位{object_name}", f"执行打开{object_name}", f"确认{object_name}已打开")
    return (f"定位{object_name}", f"执行{label}", f"确认{object_name}操作完成")


def _task_id(name: str) -> str:
    return next(task_id for task_id, task_name in TASKS if task_name == name)


def load_vla_action_templates() -> dict:
    candidates: list[dict] = []
    for index, line in enumerate(RAW_ROWS.splitlines(), start=1):
        sheet, task_name, object_name, label, row = line.split("|")
        candidates.append({
            "task_id": _task_id(task_name), "object_id": f"vla_{index:03d}",
            "object_name_zh": object_name, "source_operation_label": label,
            "match_keywords": [
                object_name,
                *[
                    part
                    for part in label.replace("：", "→").replace("，", "→").split("→")
                    if len(part) >= 2
                ],
                *SOURCE_OPERATION_ALIASES.get((object_name, label), ()),
            ],
            "steps": list(_steps(object_name, label)),
            "source_row": {"sheet": sheet, "row": int(row)},
        })
    return {"format": "vla_action_templates_v1", "tasks": [{"id": key, "name_zh": value} for key, value in TASKS], "candidates": candidates}


def validate_vla_action_templates(catalogue: Mapping[str, object]) -> list[str]:
    tasks = catalogue.get("tasks", [])
    candidates = catalogue.get("candidates", [])
    errors: list[str] = []
    if len(tasks) != 10:
        errors.append(f"expected 10 tasks, got {len(tasks)}")
    if len({row.get("object_id") for row in candidates}) != 82:
        errors.append("expected 82 unique objects")
    for index, row in enumerate(candidates):
        steps = row.get("steps", [])
        if len(steps) < 3 or len(set(steps)) != len(steps) or any(not step for step in steps):
            errors.append(f"invalid steps at candidate {index}")
    return errors


def _candidate(row: Mapping[str, object]) -> VlaActionCandidate:
    source = row["source_row"]
    return VlaActionCandidate(row["task_id"], row["object_id"], row["object_name_zh"], row["source_operation_label"], tuple(row["match_keywords"]), tuple(row["steps"]), source["sheet"], source["row"])


def select_vla_action_template(instruction: str, *, task: str | None = None, object_id: str | None = None) -> VlaActionCandidate | None:
    candidates = [_candidate(row) for row in load_vla_action_templates()["candidates"]]
    generated_goal_matches = [
        (len(goal), item)
        for item in candidates
        if (task is None or item.task_id == task)
        and (
            goal := naturalize_vla_goal({
                "object_id": item.object_id,
                "object_name_zh": item.object_name_zh,
                "source_operation_label": item.source_operation_label,
            })
        ) in instruction
    ]
    if generated_goal_matches:
        return max(generated_goal_matches, key=lambda pair: pair[0])[1]
    natural_matches = [
        (len(alias), item)
        for item in candidates
        for alias in NATURAL_MATCH_ALIASES.get(item.object_id, ())
        if alias in instruction and (task is None or item.task_id == task)
    ]
    if natural_matches:
        return max(natural_matches, key=lambda pair: pair[0])[1]
    if task is None:
        named_tasks = [task_id for task_id, name in TASKS if name in instruction]
        if len(named_tasks) == 1:
            task = named_tasks[0]
    matching = [item for item in candidates if item.object_name_zh in instruction or item.object_id == object_id]
    if task:
        task_matching = [item for item in matching if item.task_id == task]
        matching = task_matching or matching
    if not matching:
        return None
    exact_operation = [
        item for item in matching if item.source_operation_label in instruction
    ]
    if exact_operation:
        without_operations = instruction
        for item in exact_operation:
            without_operations = without_operations.replace(item.source_operation_label, "")
        return max(
            exact_operation,
            key=lambda item: (
                int(item.object_name_zh in without_operations),
                len(item.object_name_zh),
                len(item.source_operation_label),
                item.source_sheet,
                item.source_row,
            ),
        )
    if object_id is None and task is None:
        matching = [
            item
            for item in matching
            if any(
                keyword != item.object_name_zh and keyword in instruction
                for keyword in item.match_keywords
            )
        ]
        if not matching:
            return None
    def score(item: VlaActionCandidate) -> int:
        return sum(len(keyword) for keyword in item.match_keywords if keyword in instruction)
    return max(sorted(matching, key=lambda item: (item.source_sheet, item.source_row)), key=score)


def template_actions(candidate: VlaActionCandidate) -> list[str]:
    return [f"template_step({step})" for step in candidate.steps]
