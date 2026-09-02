"""Deterministic spoken-language data for the detailed action catalogue."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.manual_instruction_entry import OBJECT_SYNONYMS


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "datasets" / "detailed_action_semantic_train.json"
DEV_PATH = ROOT / "datasets" / "detailed_action_semantic_dev.json"

_SKELETONS = (
    "expression_skeleton_v2:direct",
    "expression_skeleton_v2:result",
    "expression_skeleton_v2:conditional",
    "expression_skeleton_v2:reference",
    "expression_skeleton_v2:sequence",
    "expression_skeleton_v2:collaboration",
)

# Each tuple supplies operation language, not politeness wrappers.  The six
# fields feed six different syntactic families below: direct imperative,
# desired result, conditional inspection, object reference, ordered steps,
# and coordination with related objects.
_TASK_LANGUAGE = {
    "cleaning": ("清洁干净", "完成去污并复查边角", "材质和脏污程度", "清洁任务", "清掉表面杂物", "分区擦洗并检查残留"),
    "bathroom_personal_care": ("分类整理好", "按卫生分区完成收纳", "干湿状态和交叉污染", "卫浴整理", "检查包装与洁净度", "归位到合适的干燥区域"),
    "indoor_cleaning": ("清洗后收纳", "恢复洁净并安全归位", "污渍和残留物", "室内清洁", "做彻底清洁", "晾干后分类放好"),
    "waste_disposal": ("分类丢弃", "送进正确回收区域", "材质和是否有泄漏", "垃圾分类", "密封或整理", "搬走并清洁接触处"),
    "cooking_heating": ("准备好用于烹饪", "摆到可安全使用的状态", "洁净度和使用风险", "备餐任务", "检查并备好", "配合食材完成烹饪准备"),
    "smart_cooking": ("调节到烹饪所需状态", "完成加热并安全停机", "电源和运行状态", "智能烹饪", "设置温度时间", "监测运行并检查断电"),
    "food_serving": ("摆到用餐位置", "放稳并方便取用", "洁净度和承载位置", "上餐摆放", "清洁检查", "按席位摆放并复核稳定"),
    "tableware_placement": ("清洁后摆好", "放到餐具指定位置", "残留和摆放空间", "餐具布置", "洗净沥干", "与其他餐具配套摆放"),
    "appliance_management": ("检查并调节到目标状态", "完成设备状态切换", "供电和安全联锁", "家电管理", "读取当前状态", "操作后确认响应正常"),
    "storage_open_close": ("按要求开关到位", "完成开合并保持安全", "轨道锁扣和周围障碍", "门架开合", "清除阻挡", "低速开关并确认贴合"),
    "item_delivery": ("取出后送到指定位置", "安全搬到目标台面", "包装和搬运通道", "物品递送", "稳妥取出", "搬送后放稳并复查"),
    "organizing": ("分类整理并归位", "恢复整齐易取的状态", "类别数量和目标区域", "家居整理", "盘点分类", "按频率归位并检查稳定"),
    "organizing_storage": ("装好物品并稳妥收纳", "完成分区收纳", "容量承重和洁净度", "容器收纳", "规划装载顺序", "分类装入并放稳"),
    "clothing_care": ("按护理标签处理", "护理到干燥平整", "材质颜色和污渍", "衣物护理", "读取洗护要求", "清洗或熨烫后检查状态"),
    "clothing_entryway": ("分类整理并放好", "归入对应衣物或玄关区域", "干湿洁净和类别", "衣物玄关整理", "完成检查分类", "成组归位并保持通风"),
    "laundry": ("按标签清洗护理", "洗净并正确晾干", "面料颜色和污渍", "衣物洗护", "分类预处理", "选择程序清洗后整形晾干"),
    "bedroom_service": ("整理到适合就寝的状态", "完成卧室归位布置", "位置和使用状态", "卧室服务", "检查并整理", "与寝具协同布置好"),
    "window_care": ("检查后清洁调节", "恢复洁净安全的开合状态", "开合供电和表面状态", "门窗养护", "排查障碍污渍", "清洁调节并复核安全"),
    "object_fetching": ("取来放到指定位置", "送到我方便拿取的地方", "当前位置和搬运通道", "取物任务", "定位并稳妥抓取", "搬来放稳后告诉我"),
    "security_monitoring": ("检查安全状态", "完成巡检并报告异常", "外观供电和有效状态", "安防巡检", "读取状态数值", "对照阈值检查并上报"),
    "elderly_assistance": ("检查并准备好用于护理", "放到安全易取的护理位置", "有效期洁净度和状态", "护理协助", "核对使用条件", "准备妥当并提醒注意事项"),
    "maintenance_management": ("照料并检查维护状态", "完成养护并排除风险", "环境和当前状态", "日常养护", "检查维护需求", "实施照料后复核安全"),
    "workspace_service": ("整理摆放到工作位置", "布置成可直接办公的状态", "电源线缆和目标空间", "工作区整理", "检查并清出位置", "与桌面设备配合摆放"),
    "entertainment_service": ("准备并调节好", "设置成可直接娱乐使用", "电源连接和运行状态", "娱乐设备准备", "检查并启动设置", "和相关设备联动调好"),
    "indoor_installation": ("安装摆放并固定好", "完成安装且保持稳固", "接口承重和周围风险", "室内安装", "对准位置并连接", "固定后做稳定检查"),
}

_DEV_TASK_ACTION = {
    "cleaning": "把{name}上的污垢彻底擦净",
    "bathroom_personal_care": "将{name}依卫生类别安置妥当",
    "indoor_cleaning": "洗净{name}并晾透再归置",
    "waste_disposal": "把{name}投进适合的回收类别",
    "cooking_heating": "将{name}备妥以便下厨",
    "smart_cooking": "用{name}把食材烹制好后关机",
    "food_serving": "将{name}稳稳置于用餐区",
    "tableware_placement": "洗妥{name}后放进餐位",
    "appliance_management": "让{name}进入所需工作模式",
    "storage_open_close": "把{name}平顺开合到位",
    "item_delivery": "把{name}从原处拿到目标台面",
    "organizing": "给{name}重新分组摆齐",
    "organizing_storage": "把物件有序装入{name}",
    "clothing_care": "依面料要求打理{name}",
    "clothing_entryway": "把{name}按玄关类别放妥",
    "laundry": "依洗标洗好{name}再自然晾透",
    "bedroom_service": "将{name}布置成睡前所需状态",
    "window_care": "擦去{name}的积尘并试验启闭是否顺滑",
    "object_fetching": "找到{name}并拿来给我",
    "security_monitoring": "确认{name}是否可靠，有异常就报告",
    "elderly_assistance": "把{name}备在护理时容易拿到的位置",
    "maintenance_management": "给{name}做日常照看并排除隐患",
    "workspace_service": "把{name}布置到便于办公的位置",
    "entertainment_service": "调好{name}以便马上休闲",
    "indoor_installation": "把{name}装牢并测试稳定",
}

_DEV_OBJECT_ALIASES = {
    "复合地板": "层压板面", "大理石地面": "石材地坪", "扫地机器人": "自动吸尘机",
    "塑胶地板": "PVC地面", "木地板": "木质地面", "卫浴搁板": "洗手间置物架",
    "卫生巾盒": "经期用品盒", "固体香皂": "皂块", "洗涤剂": "洗衣液",
    "皂液器": "洗手液泵", "水龙头": "龙头", "马克杯": "带把杯",
    "清洁刷": "刷洗工具", "清洁喷雾瓶": "去污喷壶", "垃圾桶": "废物筒",
    "饮料罐": "饮品铁罐", "玻璃瓶": "透明瓶器", "塑料袋": "塑料提袋",
    "可回收瓶": "回收饮品瓶", "削皮刀": "果蔬刨刀", "平底锅": "煎锅",
    "打蛋器": "蛋液搅拌棒", "擀面杖": "面团滚杖", "混合碗": "搅拌盆",
    "烤面包机下层烤盘": "吐司机底部烘盘", "烹饪托盘": "料理盘", "玻璃罐": "透明储存罐",
    "调味罐": "佐料瓶", "电磁炉": "电炉", "微波炉": "微波加热器",
    "压力锅": "高压锅", "咖啡杯": "咖啡饮具", "披萨刀": "披萨切轮",
    "洗碗刷": "餐具刷", "铝箔纸": "锡纸", "冰箱层架": "冷藏柜搁架",
    "冰箱抽屉": "冷藏柜储物屉", "冰箱门": "冷藏柜门板", "微波炉门": "微波加热器门板",
    "洗碗机上层架": "餐具清洗机顶层托架", "洗碗机门": "餐具清洗设备门板", "烤箱门": "烘烤炉门板",
    "烤面包机烤箱门": "吐司烘烤设备门板", "橱柜门": "厨柜门板", "盒装饮料": "纸盒饮品",
    "罐装物": "金属罐头", "饮料瓶": "饮品瓶", "密封保鲜盒": "密闭食物盒",
    "床头柜": "床边小柜", "收纳盒": "储物盒", "收纳箱": "储物箱",
    "收纳篮": "置物篮", "洗衣机": "洗衣设备", "棒球帽": "鸭舌帽",
    "电动窗帘": "自动帘", "卧室灯": "睡房灯具", "洗衣篮": "脏衣筐",
    "百叶窗": "百叶帘", "吸顶灯": "顶灯", "门把手": "门拉手",
    "灭火器": "消防瓶", "摄像头": "监控镜头", "卫生巾": "经期用品",
    "办公椅": "工作座椅", "文件夹": "资料夹", "显示器": "电脑屏幕",
    "笔记本": "记事本", "笔记本电脑": "便携电脑", "订书机": "装订器",
    "遥控器": "远程控制器", "数码相机": "数字相机", "相机三脚架": "摄影支架",
}

_SCENE_DEV_GOALS = {
    "MDC-001": "把多种地面区域都除尘去渍", "MDC-002": "给几类地板做日常擦洗",
    "MDC-003": "将个人洗护物分区放妥", "MDC-004": "让洗手间用品恢复整齐",
    "MDC-005": "把洁具表面彻底刷洗", "MDC-006": "洗净饭碗杯叉等器具",
    "MDC-007": "撤下用过的盘锅勺", "MDC-008": "备齐刷洗喷洒用品",
    "MDC-009": "按材质分流废弃包装", "MDC-010": "收走厨余与瓶袋垃圾",
    "MDC-011": "把下厨前的器械配齐", "MDC-012": "统筹灶具与烘烤器材做饭",
    "MDC-013": "配齐早餐所需器皿", "MDC-014": "安排午间烹调设备",
    "MDC-015": "协调加热设备制作晚饭", "MDC-016": "按席位铺陈饮食器皿",
    "MDC-017": "备妥喝水盛菜用具", "MDC-018": "将饮品容器和盘具摆稳",
    "MDC-019": "归整夹取切分与搅拌工具", "MDC-020": "备齐沥水包裹类厨具",
    "MDC-021": "归整冷藏空间及门屉", "MDC-022": "巡查厨电门板和托架",
    "MDC-023": "使厨柜屉门归位", "MDC-024": "把各类饮品送到取用处",
    "MDC-025": "按类别归置家居物件", "MDC-026": "恢复储物家具的整齐状态",
    "MDC-027": "用盒篮完成分区储藏", "MDC-028": "启动一轮衣物洗护安排",
    "MDC-029": "按面料照料裙袜正装", "MDC-030": "分类收好换季穿戴物",
    "MDC-031": "洗护衬衣鞋巾等物", "MDC-032": "把帽鞋及鞋柜归整",
    "MDC-033": "调成适合入睡的房间状态", "MDC-034": "归整床铺挂衣和脏衣区",
    "MDC-035": "将寝具摆成舒适睡前状态", "MDC-036": "清洁遮光防虫等窗部件",
    "MDC-037": "巡查玻璃窗面与边台", "MDC-038": "调节照明和空气流动",
    "MDC-039": "备齐外出随身用品", "MDC-040": "检查门口消防等防护设施",
    "MDC-041": "核验监控锁具和门口设备", "MDC-042": "备好测量与急救器材",
    "MDC-043": "把护理耗材放到易取处", "MDC-044": "养护露台植物及护栏",
    "MDC-045": "配置照明图书和便携设备开始工作", "MDC-046": "归整屏幕资料与主机区域",
    "MDC-047": "布置文具电脑等学习用品", "MDC-048": "联动影音和游戏设备休闲",
    "MDC-049": "安装装饰与摄影陈设",
}


def normalize_semantic_text(text: str) -> str:
    """Normalize text for collision and leakage checks."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def catalog_sha256(path: Path) -> str:
    """Return the digest of the exact catalogue bytes used for training."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object_phrases(entry: Mapping[str, Any]) -> list[str]:
    task = str(entry.get("task", ""))
    try:
        direct, result, condition, context, first, second = _TASK_LANGUAGE[task]
    except KeyError as error:
        raise ValueError(f"unsupported detailed action task: {task}") from error
    obj = str(entry["object_name_zh"]).strip()
    dev_name = _dev_object_name(entry)
    return [
        f"把{obj}{direct}",
        f"我希望{obj}{result}，完成后检查一下",
        f"先看看{obj}的{condition}，条件合适再{direct}",
        f"{obj}这件物品按{context}来处理，重点是{result}",
        f"先处理{obj}：{first}，然后{second}",
        _DEV_TASK_ACTION[task].format(name=dev_name),
    ]


def _dev_object_name(entry: Mapping[str, Any]) -> str:
    """Choose a natural held-out name, never reusing a long formal name."""
    formal_name = str(entry["object_name_zh"]).strip()
    for alias in OBJECT_SYNONYMS.get(str(entry.get("object_id", "")), []):
        alias = str(alias).strip()
        if (
            alias
            and alias != formal_name
            and formal_name not in alias
            and any("\u4e00" <= character <= "\u9fff" for character in alias)
        ):
            return alias
    if formal_name in _DEV_OBJECT_ALIASES:
        return _DEV_OBJECT_ALIASES[formal_name]
    if len(normalize_semantic_text(formal_name)) <= 2:
        return formal_name
    return "这个目标物"


def build_semantic_feature_lexicons(catalog: Mapping) -> tuple[
    dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]
]:
    """Build deterministic entity, task, and scene-composition lexicons."""
    entries = catalog.get("entries", [])
    entity_surfaces: dict[str, list[str]] = {}
    task_surfaces: dict[str, set[str]] = defaultdict(set)
    scene_entities: dict[str, list[str]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        label = str(entry.get("label", ""))
        if entry.get("entry_type") == "object":
            surfaces = {
                str(entry.get("object_name_zh", "")).strip(),
                _dev_object_name(entry),
                *(
                    str(value).strip()
                    for value in OBJECT_SYNONYMS.get(str(entry.get("object_id", "")), [])
                ),
            }
            entity_surfaces[label] = sorted(
                {normalize_semantic_text(value) for value in surfaces if value}
            )
            task = str(entry.get("task", ""))
            for phrase in _TASK_LANGUAGE[task]:
                normalized = normalize_semantic_text(phrase)
                if len(normalized) >= 2:
                    task_surfaces[task].add(normalized)
            for fragment in _DEV_TASK_ACTION[task].split("{name}"):
                normalized = normalize_semantic_text(fragment)
                if len(normalized) >= 2:
                    task_surfaces[task].add(normalized)
        elif entry.get("entry_type") == "scene":
            scene_entities[label] = sorted(
                str(item.get("object_label", ""))
                for item in entry.get("objects", [])
                if str(item.get("object_label", ""))
            )
    return (
        entity_surfaces,
        {task: sorted(surfaces) for task, surfaces in sorted(task_surfaces.items())},
        scene_entities,
    )


def _scene_phrases(
    entry: Mapping[str, Any], object_entries: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    command = str(entry.get("command", "")).strip()
    theme = command
    for prefix in ("帮我", "我要"):
        if theme.startswith(prefix):
            theme = theme[len(prefix) :].strip()
            break
    theme = theme.rstrip("了").strip()
    if "该" in theme:
        subject, action = theme.split("该", 1)
        if subject and action:
            theme = f"{action}{subject}"
    object_names = [
        str(item.get("object_name_zh", "")).strip()
        for item in entry.get("objects", [])
        if isinstance(item, Mapping) and str(item.get("object_name_zh", "")).strip()
    ]
    focus = "、".join(object_names[:3]) or "相关物品"
    scenario_id = str(entry.get("scenario_id", ""))
    try:
        dev_goal = _SCENE_DEV_GOALS[scenario_id]
    except KeyError as error:
        raise ValueError(f"missing held-out scene goal for {scenario_id}") from error
    dev_names = []
    for item in entry.get("objects", [])[:3]:
        referenced = object_entries.get(str(item.get("object_label", "")))
        if referenced is not None:
            dev_names.append(_dev_object_name(referenced))
    dev_focus = "、".join(dev_names) or "这些目标物"
    return [
        command,
        f"我想完成这件事：{theme}，你来安排各步骤",
        f"先检查{focus}的状态，条件合适就照这个目标执行：{theme}",
        f"这次要{theme}，相关物品按用途逐一协调",
        f"先确认环境安全，再按顺序执行：{theme}，最后复查结果",
        f"把{dev_focus}统筹起来，{dev_goal}",
    ]


def build_detailed_action_examples(catalog: Mapping) -> list[dict]:
    """Build six deterministic spoken variants for every catalogue label."""
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("catalog entries must be a non-empty list")

    examples: list[dict] = []
    labels_by_text: dict[str, str] = {}
    object_entries = {
        str(entry.get("label")): entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("entry_type") == "object"
    }
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("every catalog entry must be a mapping")
        label = str(entry.get("label", "")).strip()
        entry_type = entry.get("entry_type")
        if not label:
            raise ValueError("every catalog entry must have a label")
        if entry_type == "scene":
            phrases = _scene_phrases(entry, object_entries)
        elif entry_type == "object":
            phrases = _object_phrases(entry)
        else:
            raise ValueError(f"unsupported catalog entry type: {entry_type}")

        for source, phrase in zip(_SKELETONS, phrases):
            normalized = normalize_semantic_text(phrase)
            previous_label = labels_by_text.get(normalized)
            if previous_label is not None:
                raise ValueError(
                    "normalized semantic text collision: "
                    f"{phrase!r} belongs to both {previous_label!r} and {label!r}"
                )
            labels_by_text[normalized] = label
            examples.append({"text": phrase, "label_id": label, "source": source})
    return examples


def _stable_skeleton_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def split_detailed_action_examples(
    examples: Sequence[Mapping[str, str]],
) -> tuple[list[dict], list[dict]]:
    """Hold out one complete expression-skeleton family for development."""
    rows_by_label: dict[str, list[dict]] = defaultdict(list)
    labels_by_text: dict[str, str] = {}
    for example in examples:
        row = {
            "text": str(example["text"]),
            "label_id": str(example["label_id"]),
            "source": str(example["source"]),
        }
        normalized = normalize_semantic_text(row["text"])
        previous_label = labels_by_text.get(normalized)
        if previous_label is not None:
            raise ValueError(
                f"duplicate normalized text for {previous_label!r} and {row['label_id']!r}"
            )
        labels_by_text[normalized] = row["label_id"]
        rows_by_label[row["label_id"]].append(row)

    sources = sorted({row["source"] for rows in rows_by_label.values() for row in rows})
    if len(sources) < 2:
        raise ValueError("at least two expression skeletons are required for splitting")
    dev_source = min(sources, key=lambda source: (_stable_skeleton_digest(source), source))

    train: list[dict] = []
    dev: list[dict] = []
    for label in sorted(rows_by_label):
        rows = sorted(
            rows_by_label[label],
            key=lambda row: (row["source"], normalize_semantic_text(row["text"])),
        )
        if len(rows) < 2:
            raise ValueError(f"label needs at least two examples for splitting: {label}")
        label_dev = [row for row in rows if row["source"] == dev_source]
        label_train = [row for row in rows if row["source"] != dev_source]
        if len(label_dev) != 1 or not label_train:
            raise ValueError(f"label does not cover every expression skeleton: {label}")
        dev.extend(label_dev)
        train.extend(label_train)
    return train, dev


def write_detailed_action_splits(
    catalog: Mapping,
    *,
    train_path: Path = TRAIN_PATH,
    dev_path: Path = DEV_PATH,
) -> tuple[list[dict], list[dict]]:
    """Generate and write the deterministic train/development files."""
    train, dev = split_detailed_action_examples(build_detailed_action_examples(catalog))
    for path, rows in ((Path(train_path), train), (Path(dev_path), dev)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return train, dev
