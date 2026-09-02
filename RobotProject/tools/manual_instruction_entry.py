"""Manual natural-language instruction matching for the 15-task / 120-object scope."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.instruction_parser import parse_instruction
from tools.household_catalog import acceptance_task, tasks_for_object
from tools.operation_plans import DEFAULT_OPERATION_BY_TASK, operation_plan


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "datasets" / "standard_instruction_action_benchmark.json"
CATALOG_PATH = ROOT / "meta" / "compliance_catalog_15x120.json"

OPERATION_NAMES_ZH = {
    "care_plant": "养护花卉",
    "water_plants": "浇灌花卉",
    "operate_game_controller": "操控游戏手柄",
    "play_audio": "播放音频",
    "iron": "熨烫",
    "fold": "折叠",
    "inspect": "检查",
    "wash": "清洗",
}

PRIMITIVE_NAMES_ZH = {
    "locate": "定位", "grasp": "抓取", "fill": "接水/注水", "move": "移动",
    "water": "浇水", "place": "放回", "inspect_plant_and_pests": "检查植株和虫害",
    "remove_pests": "去除害虫", "organize": "整理", "inspect": "检查",
    "dry": "晾干", "fold": "折叠", "iron": "熨烫",
    "turn_on": "开机", "connect": "连接设备", "start_game": "启动游戏",
    "control": "操控游戏", "play": "播放内容", "adjust_volume": "调节音量",
}
PRIMITIVE_NAMES_ZH["clean"] = "\u6e05\u6d01"

OPERATION_DISPLAY_ZH = {
    "turn_on": "打开",
    "turn_off": "关闭",
    "water_plants": "给花浇水",
    "maintain": "养护",
    "wash": "清洗",
    "fetch": "拿来",
    "serve": "递送",
    "pour_water": "倒水",
    "open": "打开",
    "close": "关闭",
    "clean": "清洁",
    "inspect": "检查",
    "organize": "整理",
    "operate": "使用",
}


TASK_KEYWORDS = {
    "floor_cleaning": ["地面清洁", "扫地", "拖地", "擦地", "清洁地面", "地板清洁"],
    "organizing": ["整理", "收纳", "归纳", "摆好", "放整齐", "整理一下"],
    "smart_cooking": ["烹饪", "做饭", "煮饭", "加热", "烧水", "做菜", "煮"],
    "appliance_management": ["家电", "打开", "关闭", "开一下", "关一下", "控制", "启动"],
    "security_monitoring": ["安防", "安全", "巡检", "检查", "报警", "门锁", "摄像头"],
    "laundry": ["衣物清洗", "清洗衣物", "洗衣", "洗衣服", "洗衣物", "洗鞋", "清洗鞋"],
    "dish_washing": ["餐具清洗", "洗碗", "洗杯", "洗盘", "清洗餐具", "把杯子洗了", "洗了", "洗一下", "拿去洗"],
    "waste_disposal": ["垃圾", "丢掉", "扔掉", "回收", "处理垃圾", "垃圾处理"],
    "clothing_care": ["服装护理", "熨", "熨烫", "折叠", "护理衣服", "整理衣服"],
    "window_care": ["门窗", "窗户", "窗帘", "擦窗", "开窗", "关窗"],
    "bathroom_cleaning": ["卫浴", "厕所", "马桶", "浴室", "水槽", "淋浴", "清洁卫生间"],
    "bedroom_service": ["卧室", "铺床", "整理床", "枕头", "被子", "床头"],
    "food_serving": ["送餐", "送水", "递水", "递水杯", "拿杯子", "拿水杯", "饮水", "端菜", "上菜"],
    "object_fetching": ["取物", "拿", "拿来", "拿过来", "递给我", "取来", "找一下"],
    "elderly_assistance": ["elderly_assistance", "health care", "healthcare", "健康关怀", "助老", "测量", "体温", "血糖", "轮椅", "眼镜", "口罩"],
    "maintenance_management": ["维护管理", "维护任务"],
    "entertainment_service": ["娱乐服务", "娱乐任务"],
}

INTENT_KEYWORDS = {
    "object_fetching": ["拿给", "递给", "取来", "拿来", "拿过来", "送给"],
    "entertainment_service": ["播放", "节目", "音量", "频道", "换台", "打开电视", "关闭电视", "电视开机", "电视关机", "电视电源", "用遥控器", "游戏", "音乐", "play", "music", "game"],
    "maintenance_management": ["维护", "浇水", "浇点水", "灌溉", "过滤器", "滤网", "保养", "检查设备", "害虫", "water", "fill"],
    "floor_cleaning": ["扫地", "拖地", "擦地", "清洁木地板", "清洁地板"],
    "dish_washing": ["洗杯", "洗碗", "洗盘", "杯子洗", "餐具清洗"],
    "bathroom_cleaning": ["清洁卫生间", "刷马桶", "清洁浴室"],
}


# High-confidence verb phrases are resolved before the broad keyword fallback.
# The object may be implicit when the phrase itself identifies it (for example,
# “倒杯水” conventionally implies a drinking cup).
INTENT_PHRASE_RULES = (
    {
        "task": "food_serving",
        "object": "water_cup",
        "operation": "pour_water",
        "target": "person",
        "phrases": ("给我倒杯水", "帮我倒水", "倒点水过来", "我想喝水"),
        "priority": 220,
    },
    {
        "task": "appliance_management",
        "object": "reading_lamp",
        "operation": "turn_off",
        "target": "none",
        "phrases": ("把阅读灯关掉", "关闭阅读灯", "熄灭阅读灯", "关阅读灯", "睡前把阅读灯关了"),
        "priority": 220,
    },
    {
        "task": "appliance_management",
        "object": "reading_lamp",
        "operation": "turn_on",
        "target": "none",
        "phrases": ("把阅读灯打开", "打开阅读灯", "开启阅读灯", "开阅读灯"),
        "priority": 220,
    },
    {
        "task": "maintenance_management",
        "object": "flower_pot",
        "operation": "water_plants",
        "target": "flower_pot",
        "phrases": ("帮我浇花", "给花浇水", "浇花", "给盆栽浇水", "花有点干，帮我浇点水"),
        "priority": 220,
    },
    {
        "task": "maintenance_management",
        "object": "flower_pot",
        "operation": "care_plant",
        "target": "flower_pot",
        "phrases": ("照顾花", "照料花", "养护花", "照顾盆栽"),
        "priority": 220,
    },
    {
        "task": "entertainment_service",
        "object": "speaker",
        "operation": "play_audio",
        "target": "audio_source",
        "phrases": ("打开音响", "打开扬声器", "播放音响", "播放音乐"),
        "priority": 220,
    },
    {
        "task": "appliance_management",
        "object": "ceiling_light",
        "operation": "turn_on",
        "target": "none",
        "phrases": ("把灯打开", "打开灯", "开灯"),
        "priority": 195,
    },
    {
        "task": "appliance_management",
        "object": "ceiling_light",
        "operation": "turn_off",
        "target": "none",
        "phrases": ("把灯关掉", "关闭灯", "关灯"),
        "priority": 195,
    },
    {
        "task": "entertainment_service",
        "object": "television",
        "operation": "turn_on",
        "target": "none",
        "phrases": ("把电视打开", "打开电视", "电视开机", "开电视"),
        "exclude": ("遥控器",),
        "priority": 194,
    },
    {
        "task": "appliance_management",
        "object": "fan",
        "operation": "turn_on",
        "target": "none",
        "phrases": ("把风扇打开", "打开风扇", "开风扇"),
        "priority": 193,
    },
    {
        "task": "appliance_management",
        "object": "fan",
        "operation": "turn_off",
        "target": "none",
        "phrases": ("把风扇关掉", "关闭风扇", "关风扇"),
        "priority": 193,
    },
    {
        "task": "entertainment_service",
        "object": "television",
        "operation": "turn_off",
        "target": "none",
        "phrases": ("把电视关掉", "关闭电视", "电视关机", "关电视", "电视不用看了，帮我关掉"),
        "exclude": ("遥控器",),
        "priority": 194,
    },
    {
        "task": "maintenance_management",
        "object": "air_conditioner",
        "operation": "maintain",
        "target": "none",
        "phrases": ("检查空调滤网", "清洁空调滤网", "更换空调滤网", "维护空调", "保养空调"),
        "priority": 200,
    },
    {
        "task": "appliance_management",
        "object": "air_conditioner",
        "operation": "turn_on",
        "target": "none",
        "phrases": ("把空调打开", "打开空调", "启动空调", "开空调"),
        "priority": 190,
    },
    {
        "task": "appliance_management",
        "object": "air_conditioner",
        "operation": "turn_off",
        "target": "none",
        "phrases": ("把空调关掉", "关闭空调", "关空调"),
        "priority": 190,
    },
    {
        "task": "appliance_management",
        "object": "electric_curtain",
        "operation": "open",
        "target": "none",
        "phrases": ("打开电动窗帘", "开启电动窗帘", "把电动窗帘打开"),
        "priority": 225,
    },
    {
        "task": "appliance_management",
        "object": "electric_curtain",
        "operation": "close",
        "target": "none",
        "phrases": ("关闭电动窗帘", "关上电动窗帘", "把电动窗帘关闭"),
        "priority": 225,
    },
    {
        "task": "window_care",
        "object": "window",
        "operation": "open",
        "target": "none",
        "phrases": ("把窗户打开", "打开窗户", "开窗户", "开窗"),
        "priority": 185,
    },
    {
        "task": "window_care",
        "object": "curtain",
        "operation": "open",
        "target": "none",
        "phrases": ("打开窗帘", "拉开窗帘"),
        "priority": 186,
    },
    {
        "task": "window_care",
        "object": "blind",
        "operation": "open",
        "target": "none",
        "phrases": ("打开百叶窗", "拉开百叶窗"),
        "priority": 186,
    },
    {
        "task": "window_care",
        "object": "window",
        "operation": "close",
        "target": "none",
        "phrases": ("把窗户关上", "关闭窗户", "关窗户", "关窗"),
        "priority": 185,
    },
    {
        "task": "window_care",
        "object": "window",
        "operation": "clean",
        "target": "none",
        "phrases": ("擦窗户", "清洁窗户", "洗窗户"),
        "priority": 185,
    },
    {
        "task": "food_serving",
        "object": "water_cup",
        "operation": "pour_water",
        "target": "person",
        "phrases": ("帮我倒杯水", "倒杯水", "倒一杯水", "给我倒水", "接一杯水", "接杯水"),
        "priority": 180,
    },
    {
        "task": "food_serving",
        "object": "water_cup",
        "operation": "serve",
        "target": "person",
        "phrases": ("给我送杯水", "递杯水给我", "送杯水"),
        "priority": 180,
    },
    {
        "task": "dish_washing",
        "object": "cup",
        "operation": "wash",
        "target": "sink",
        "phrases": ("把水杯洗一下", "洗水杯", "把杯子洗一下", "洗杯子"),
        "priority": 170,
    },
    {
        "task": "object_fetching",
        "object": "water_cup",
        "operation": "fetch",
        "target": "person",
        "phrases": ("把水杯拿给我", "水杯拿给我", "拿水杯给我", "取水杯"),
        "priority": 160,
    },
)

CROSS_INTENT_ACTION_PLANS = {
    ("maintenance_management", "air_conditioner"): {
        "target": "none",
        "actions": [
            "locate(air_conditioner)",
            "inspect(air_conditioner)",
            "clean(air_conditioner)",
            "inspect(air_conditioner)",
        ],
    },
    ("object_fetching", "water_cup"): {
        "target": "person",
        "actions": [
            "locate(water_cup)",
            "grasp(water_cup)",
            "move(person)",
            "release(water_cup)",
        ],
    },
}


OBJECT_SYNONYMS = {
    "carpet": ["carpet", "地毯"],
    "wood_floor": ["wood_floor", "木地板"],
    "tile_floor": ["tile_floor", "瓷砖", "瓷砖地"],
    "marble_floor": ["marble_floor", "大理石地面"],
    "laminate_floor": ["laminate_floor", "复合地板"],
    "vinyl_floor": ["vinyl_floor", "塑胶地板"],
    "floor_mat": ["floor_mat", "地垫", "脚垫"],
    "robot_vacuum": ["robot_vacuum", "扫地机器人"],
    "storage_box": ["storage_box", "收纳箱", "储物箱"],
    "bookshelf": ["bookshelf", "书架"],
    "wardrobe": ["wardrobe", "衣柜"],
    "drawer": ["drawer", "抽屉"],
    "cabinet": ["cabinet", "柜子", "橱柜"],
    "desk": ["desk", "书桌", "桌子"],
    "coffee_table": ["coffee_table", "茶几"],
    "toy": ["toy", "玩具"],
    "rice_cooker": ["rice_cooker", "电饭煲", "饭煲"],
    "induction_cooker": ["induction_cooker", "电磁炉"],
    "oven": ["oven", "烤箱"],
    "microwave": ["microwave", "微波炉"],
    "frying_pan": ["frying_pan", "平底锅", "煎锅"],
    "electric_kettle": ["electric_kettle", "电热水壶", "水壶"],
    "blender": ["blender", "搅拌机", "破壁机"],
    "pressure_cooker": ["pressure_cooker", "压力锅"],
    "ceiling_light": ["ceiling_light", "吸顶灯"],
    "reading_lamp": ["reading_lamp", "阅读灯", "台灯"],
    "television": ["television", "电视", "电视机"],
    "air_conditioner": ["air_conditioner", "空调"],
    "refrigerator": ["refrigerator", "冰箱"],
    "washing_machine": ["washing_machine", "洗衣机"],
    "electric_curtain": ["electric_curtain", "电动窗帘"],
    "fan": ["fan", "风扇"],
    "smart_lock": ["smart_lock", "智能门锁", "门锁"],
    "security_camera": ["security_camera", "摄像头", "监控摄像头"],
    "fire_alarm": ["fire_alarm", "火灾报警器", "报警器"],
    "fire_extinguisher": ["fire_extinguisher", "灭火器"],
    "doorknob": ["doorknob", "门把手"],
    "padlock": ["padlock", "挂锁"],
    "doorbell_button": ["doorbell_button", "doorbell", "门铃", "门铃按钮"],
    "video_doorbell": ["video_doorbell", "可视门铃", "视频门铃"],
    "child_clothing": ["child_clothing", "儿童服装", "儿童衣物", "衣物", "衣服"],
    "shoes": ["shoes", "鞋", "鞋子"],
    "cotton_coat": ["cotton_coat", "棉衣", "棉服"],
    "down_jacket": ["down_jacket", "羽绒服", "羽绒制品"],
    "wool_garment": ["wool_garment", "羊毛衫", "羊毛衣物"],
    "shirt": ["shirt", "衬衫"],
    "pants": ["pants", "裤子"],
    "towel": ["towel", "毛巾"],
    "cup": ["cup", "杯子", "水杯"],
    "mug": ["mug", "马克杯"],
    "bowl": ["bowl", "碗"],
    "plate": ["plate", "盘子"],
    "spoon": ["spoon", "勺子", "汤勺"],
    "fork": ["fork", "叉子"],
    "knife": ["knife", "刀", "刀具"],
    "pot": ["pot", "锅"],
    "trash_bin": ["trash_bin", "垃圾桶"],
    "recyclable_bottle": ["recyclable_bottle", "可回收瓶"],
    "food": ["food", "食物", "食品", "饭菜", "厨余垃圾", "食物垃圾"],
    "plastic_bag": ["plastic_bag", "塑料袋"],
    "cardboard_box": ["cardboard_box", "纸箱"],
    "aluminum_can": ["aluminum_can", "易拉罐", "铝罐"],
    "glass_bottle": ["glass_bottle", "玻璃瓶"],
    "battery": ["battery", "电池"],
    "dress": ["dress", "连衣裙", "裙子"],
    "suit": ["suit", "西装"],
    "sweater": ["sweater", "毛衣"],
    "skirt": ["skirt", "短裙", "半身裙"],
    "socks": ["socks", "袜子"],
    "scarf": ["scarf", "围巾"],
    "bed_sheet": ["bed_sheet", "床单"],
    "iron": ["iron", "熨斗"],
    "window": ["window", "窗户", "窗"],
    "window_glass": ["window_glass", "窗玻璃", "玻璃窗"],
    "window_frame": ["window_frame", "窗框"],
    "curtain": ["curtain", "窗帘"],
    "blind": ["blind", "百叶窗"],
    "screen_window": ["screen_window", "纱窗"],
    "balcony_railing": ["balcony_railing", "阳台栏杆", "栏杆", "护栏", "阳台门"],
    "window_sill": ["window_sill", "窗台"],
    "toilet": ["toilet", "马桶", "厕所"],
    "sink": ["sink", "水槽", "洗手池"],
    "bathtub": ["bathtub", "浴缸"],
    "shower": ["shower", "淋浴"],
    "mirror": ["mirror", "镜子"],
    "bathroom_floor": ["bathroom_floor", "浴室地面", "卫生间地面"],
    "faucet": ["faucet", "水龙头"],
    "soap": ["soap", "肥皂", "香皂"],
    "bed": ["bed", "床"],
    "pillow": ["pillow", "枕头"],
    "quilt": ["quilt", "被子"],
    "mattress": ["mattress", "床垫"],
    "nightstand": ["nightstand", "床头柜"],
    "bedroom_lamp": ["bedroom_lamp", "卧室灯"],
    "clothes_hanger": ["clothes_hanger", "衣架"],
    "laundry_basket": ["laundry_basket", "洗衣篮"],
    "water_cup": ["water_cup", "饮水杯", "水杯", "杯子"],
    "wine_glass": ["wine_glass", "酒杯", "红酒杯"],
    "bottle": ["bottle", "瓶子"],
    "food_tray": ["food_tray", "餐盘", "托盘"],
    "dinner_plate": ["dinner_plate", "餐碟"],
    "serving_bowl": ["serving_bowl", "汤碗"],
    "chopsticks": ["chopsticks", "筷子"],
    "thermos": ["thermos", "保温杯", "热水瓶"],
    "remote_control": ["remote_control", "遥控器"],
    "mobile_phone": ["mobile_phone", "手机"],
    "laptop": ["laptop", "笔记本电脑", "电脑"],
    "book": ["book", "书", "书本"],
    "medicine_box": ["medicine_box", "药盒", "药箱"],
    "walking_cane": ["walking_cane", "拐杖"],
    "keys": ["keys", "钥匙"],
    "wallet": ["wallet", "钱包"],
    "smart_scale": ["smart_scale", "智能体重秤", "体重秤"],
    "thermometer": ["thermometer", "温度计", "体温计"],
    "bandage": ["bandage", "绷带"],
    "glucose_meter": ["glucose_meter", "血糖仪"],
    "wheelchair": ["wheelchair", "轮椅"],
    "spectacles": ["spectacles", "眼镜"],
    "medical_mask": ["medical_mask", "口罩", "医用口罩"],
    "medical_glove": ["medical_glove", "医用手套", "手套"],
}


OBJECT_SYNONYMS.update({
    "blood_pressure_monitor": ["blood_pressure_monitor", "血压计", "血压仪"],
    "first_aid_kit": ["first_aid_kit", "急救箱", "急救包"],
    "sanitary_pad": ["sanitary_pad", "卫生巾"],
    "spectacles": ["spectacles", "眼镜"],
    "flower_pot": ["flower_pot", "flower pot", "花盆", "盆栽"],
    "watering_can": ["watering_can", "watering can", "浇水壶", "洒水壶", "水壶"],
    "game_controller": ["game_controller", "game controller", "controller", "游戏手柄", "手柄"],
    "speaker": ["speaker", "音箱", "扬声器"],
})

DEFAULT_OBJECT_BY_TASK = {
    "floor_cleaning": "carpet",
    "organizing": "storage_box",
    "smart_cooking": "rice_cooker",
    "appliance_management": "ceiling_light",
    "security_monitoring": "smart_lock",
    "laundry": "child_clothing",
    "dish_washing": "cup",
    "waste_disposal": "trash_bin",
    "clothing_care": "dress",
    "window_care": "window",
    "bathroom_cleaning": "toilet",
    "bedroom_service": "bed",
    "food_serving": "water_cup",
    "object_fetching": "remote_control",
    "elderly_assistance": "blood_pressure_monitor",
}


class NonExecutableInstructionError(ValueError):
    """Known semantics that cannot yet produce a canonical action plan."""


class ClarificationRequiredError(ValueError):
    """The instruction maps to multiple canonical task-object relations."""


def _clarification_error(candidates: list[str]) -> ClarificationRequiredError:
    return ClarificationRequiredError(
        "请说明具体要做什么，例如“清洗杯子”或“把水杯拿给我”；候选任务："
        + ", ".join(candidates)
    )


@dataclass(frozen=True)
class ManualInstruction:
    normalized_instruction: str
    task: str
    object: str
    target: str
    operation: str
    used_fallback: bool
    note: str
    confidence: int = 0
    matched_terms: list[str] = field(default_factory=list)
    is_executable: bool = True
    is_canonical_object: bool = True


def load_benchmark_records(path: Path = BENCHMARK_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact(text: str) -> str:
    return text.strip().lower().replace(" ", "").replace("_", "")


def _display_object(object_id: str) -> str:
    """Return a Chinese object name when a known synonym provides one."""
    for alias in OBJECT_SYNONYMS.get(object_id, []):
        if re.search(r"[\u4e00-\u9fff]", alias):
            return alias
    return object_id.replace("_", " ")


def describe_resolved_intent(resolved: Mapping[str, object]) -> str:
    """Describe parsed intent in natural Chinese without exposing internal IDs."""
    operation = str(resolved["operation"])
    object_name = _display_object(str(resolved["object"]))
    if operation == "water_plants":
        return "给花浇水"
    if operation == "pour_water":
        return "倒一杯水"
    return f"{OPERATION_DISPLAY_ZH.get(operation, operation)}{object_name}"


def _action_parts(action: str) -> tuple[str, list[str]] | None:
    match = re.fullmatch(r"([a-z_]+)\((.*)\)", action)
    if not match:
        return None
    primitive, raw_arguments = match.groups()
    return primitive, [part.strip() for part in raw_arguments.split(",") if part.strip()]


def describe_actions(actions: Sequence[str]) -> list[str]:
    """Translate known action primitives for the default user-facing result."""
    result: list[str] = []
    last_power_action: str | None = None
    for action in actions:
        parts = _action_parts(action)
        if parts is None:
            result.append(action)
            continue
        primitive, arguments = parts
        if primitive == "template_step":
            result.append(arguments[0] if arguments else action)
            continue
        object_name = _display_object(arguments[0]) if arguments else "目标"
        if primitive == "locate":
            text = f"找到{object_name}"
        elif primitive == "turn_on":
            text = f"打开{object_name}"
            last_power_action = primitive
        elif primitive == "turn_off":
            text = f"关闭{object_name}"
            last_power_action = primitive
        elif primitive == "confirm_state":
            text = "确认已经关闭" if last_power_action == "turn_off" else "确认已经开启" if last_power_action == "turn_on" else "确认当前状态"
        elif primitive == "grasp":
            text = f"拿起{object_name}"
        elif primitive == "move":
            text = "送到用户身边" if arguments and arguments[0] == "person" else f"移到{object_name}旁"
        elif primitive in {"release", "place"}:
            text = f"放好{object_name}"
        elif primitive == "wash":
            text = f"清洗{object_name}"
        elif primitive == "rinse":
            text = f"冲洗{object_name}"
        elif primitive == "dry":
            text = f"晾干{object_name}"
        elif primitive == "water":
            text = f"给{object_name}浇水"
        elif primitive in {"clean", "inspect", "organize"}:
            text = f"{PRIMITIVE_NAMES_ZH[primitive]}{object_name}"
        else:
            text = action
        result.append(text)
    return result


def _match_intent_phrase(text: str) -> tuple[dict | None, list[str]]:
    compact = _compact(text)
    matches: list[tuple[int, int, dict, str]] = []
    for rule in INTENT_PHRASE_RULES:
        if any(_compact(term) in compact for term in rule.get("exclude", ())):
            continue
        for phrase in rule["phrases"]:
            compact_phrase = _compact(phrase)
            if compact_phrase in compact:
                matches.append((rule["priority"], len(compact_phrase), rule, phrase))
    if not matches:
        return None, []
    _, _, rule, phrase = max(matches, key=lambda item: (item[0], item[1]))
    return rule, [phrase]


def _infer_operation(text: str, task: str, obj: str) -> str:
    compact = _compact(text)
    if task == "cleaning" and any(
        term in compact for term in ("清洁", "清理", "洗净", "弄干净", "擦干净")
    ):
        return "clean"
    if task == "clothing_care":
        if obj == "iron" and any(term in compact for term in ("检查", "检测", "查看")):
            return "inspect"
        if "熨烫" in compact or (obj != "iron" and compact.startswith("熨")):
            return "iron"
        if any(term in compact for term in ("折叠", "叠好")):
            return "fold"
        return "organize"
    if task == "elderly_assistance" and (
        obj in {"blood_pressure_monitor", "thermometer"}
        or any(term in compact for term in ("检查", "检测", "测量", "查看"))
    ):
        return "inspect"
    if task == "maintenance_management" and obj == "flower_pot":
        if any(term in compact for term in ("浇花", "浇水", "浇点水", "灌溉")):
            return "water_plants"
        return "maintain"
    if obj == "robot_vacuum" and "启动" in compact:
        return "robot_clean"
    if task == "security_monitoring" and any(term in compact for term in ("锁好", "上锁", "锁上")):
        return "lock"
    if task == "security_monitoring" and any(term in compact for term in ("按下", "按门铃")):
        return "press"
    if obj == "bedroom_lamp":
        if compact.startswith("关") or any(term in compact for term in ("关闭", "关灯")):
            return "turn_off"
        if compact.startswith("开") or any(term in compact for term in ("打开", "开灯")):
            return "turn_on"
    if task == "window_care":
        if any(term in compact for term in ("关闭", "关窗", "关上")):
            return "close"
        if any(term in compact for term in ("打开", "开窗")):
            return "open"
        if any(term in compact for term in ("擦", "清洁", "洗")):
            return "clean"
    if task == "appliance_management":
        if compact.startswith("关") or any(term in compact for term in ("关闭", "关掉", "关机")):
            return "turn_off"
        if compact.startswith("开") or any(term in compact for term in ("打开", "启动", "开机")):
            return "turn_on"
    return DEFAULT_OPERATION_BY_TASK.get(task, "operate")


def _catalog_indexes() -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    catalog = load_catalog()
    object_to_task: dict[str, list[str]] = {}
    task_names = {}
    for task in catalog["tasks"]:
        task_names[task["id"]] = task["name_zh"]
        for obj in task["objects"]:
            object_to_task.setdefault(obj, []).append(task["id"])
    canonical = load_catalog()
    for task in canonical.get("acceptance_tasks", []):
        task_names[task["id"]] = task["name_zh"]
    for obj in OBJECT_SYNONYMS:
        members = list(tasks_for_object(obj))
        if members:
            legacy = object_to_task.get(obj, [])
            object_to_task[obj] = list(dict.fromkeys([*legacy, *members]))
    return {obj: tuple(tasks) for obj, tasks in object_to_task.items()}, task_names


def _target_for(task: str, obj: str, records: list[dict]) -> str:
    for record in records:
        if record["task"] == task and record["object"] == obj:
            return record["target"]
    for record in records:
        if record["task"] == task:
            return record["target"]
    return "none"


def _explicit_target(text: str) -> str | None:
    compact = _compact(text)
    if any(term in compact for term in ("桌上", "桌子上", "餐桌上", "放桌")):
        return "table"
    if any(term in compact for term in ("水槽", "洗手池")):
        return "sink"
    if any(term in compact for term in ("衣柜", "衣橱")):
        return "wardrobe"
    return None
def _match_intent(text: str) -> tuple[str | None, int, list[str]]:
    compact = _compact(text)
    matches = []
    for task, keywords in INTENT_KEYWORDS.items():
        terms = [term for term in keywords if _compact(term) in compact]
        if terms:
            matches.append((sum(len(_compact(term)) for term in terms), task, terms))
    if not matches:
        return None, 0, []
    score, task, terms = max(matches)
    return task, score, terms


def _match_explicit_task(text: str, task_names: dict[str, str]) -> tuple[str | None, list[str]]:
    compact = _compact(text)
    normalized = text.strip().lower()
    matches: list[tuple[int, int, str, list[str]]] = []
    for task, name in task_names.items():
        name_terms = [name] if name and _compact(name) in compact else []
        task_terms = [task] if re.search(
            rf"(?<![a-z0-9_]){re.escape(task.lower())}(?![a-z0-9_])",
            normalized,
        ) else []
        terms = [*name_terms, *task_terms]
        if terms:
            matches.append((1 if name_terms else 0, max(len(_compact(term)) for term in terms), task, terms))
    if not matches:
        return None, []
    _, _, task, terms = max(matches, key=lambda item: (item[0], item[1]))
    return task, terms


def _match_task(text: str, task_names: dict[str, str]) -> tuple[str | None, int, list[str]]:
    compact = _compact(text)
    best_task = None
    best_score = 0
    best_terms: list[str] = []
    for task, keywords in TASK_KEYWORDS.items():
        terms = [term for term in [task, task_names.get(task, ""), *keywords] if term and _compact(term) in compact]
        score = sum(max(2, len(_compact(term))) for term in terms)
        if score > best_score:
            best_task = task
            best_score = score
            best_terms = terms
    return best_task, best_score, best_terms


def _match_object(text: str) -> tuple[str | None, int, list[str]]:
    compact = _compact(text)
    best_obj = None
    best_score = 0
    best_specificity = 0
    best_terms: list[str] = []
    for obj, aliases in OBJECT_SYNONYMS.items():
        all_aliases = [obj, *aliases]
        terms = [term for term in all_aliases if term and _compact(term) in compact]
        score = sum(max(2, len(_compact(term))) for term in terms)
        specificity = max((len(_compact(term)) for term in terms), default=0)
        if score > best_score or (score == best_score and specificity > best_specificity):
            best_obj = obj
            best_score = score
            best_specificity = specificity
            best_terms = terms
    return best_obj, best_score, best_terms


def _top_object_task_candidates(
    text: str, object_to_task: dict[str, tuple[str, ...]]
) -> tuple[list[str], dict[str, tuple[str, int, list[str]]]]:
    """Keep every equally strong task candidate instead of relying on dict order."""
    compact = _compact(text)
    by_task: dict[str, tuple[str, int, list[str]]] = {}
    representatives: dict[str, str] = {}
    for obj, aliases in OBJECT_SYNONYMS.items():
        terms = [term for term in [obj, *aliases] if term and _compact(term) in compact]
        score = sum(max(2, len(_compact(term))) for term in terms)
        if not score:
            continue
        for task in object_to_task.get(obj, ()):
            canonical = acceptance_task(task)
            current = by_task.get(canonical)
            specificity = max((len(_compact(term)) for term in terms), default=0)
            current_specificity = (
                max((len(_compact(term)) for term in current[2]), default=0)
                if current else 0
            )
            if current is None or score > current[1] or (
                score == current[1] and specificity > current_specificity
            ):
                by_task[canonical] = (obj, score, terms)
                representatives[canonical] = task
    if not by_task:
        return [], {}
    top_score = max(item[1] for item in by_task.values())
    top = [canonical for canonical, item in by_task.items() if item[1] == top_score]
    display = [representatives[canonical] for canonical in top]
    return display, {representatives[key]: by_task[key] for key in top}


def _match_object_for_task(
    text: str,
    task: str,
    object_to_task: dict[str, tuple[str, ...]],
) -> tuple[str | None, int, list[str]]:
    compact = _compact(text)
    best_obj = None
    best_score = 0
    best_specificity = 0
    best_terms: list[str] = []
    for obj, aliases in OBJECT_SYNONYMS.items():
        if task not in object_to_task.get(obj, ()) and acceptance_task(task) not in object_to_task.get(obj, ()):
            continue
        all_aliases = [obj, *aliases]
        terms = [term for term in all_aliases if term and _compact(term) in compact]
        score = sum(max(2, len(_compact(term))) for term in terms)
        specificity = max((len(_compact(term)) for term in terms), default=0)
        if score > best_score or (score == best_score and specificity > best_specificity):
            best_obj = obj
            best_score = score
            best_specificity = specificity
            best_terms = terms
    return best_obj, best_score, best_terms


def resolve_manual_instruction(
    instruction: str,
    *,
    benchmark_records: list[dict] | None = None,
) -> ManualInstruction:
    """Resolve free-form user text into one canonical benchmark operation."""
    normalized = instruction.strip()
    if not normalized:
        raise ValueError("请输入一条任务指令")
    records = benchmark_records or load_benchmark_records()
    object_to_task, task_names = _catalog_indexes()

    explicit_task, explicit_terms = _match_explicit_task(normalized, task_names)
    phrase_rule, phrase_terms = _match_intent_phrase(normalized)
    if phrase_rule:
        selected_task = (
            phrase_rule["task"]
            if explicit_task == "cleaning"
            else explicit_task or phrase_rule["task"]
        )
        return ManualInstruction(
            normalized_instruction=normalized,
            task=acceptance_task(selected_task),
            object=phrase_rule["object"],
            target=phrase_rule["target"],
            operation=phrase_rule["operation"],
            used_fallback=False,
            note="按高置信度动作短语进行多意图消歧",
            confidence=phrase_rule["priority"],
            matched_terms=[*explicit_terms, *phrase_terms],
        )

    # Explicit task names win first; otherwise action intent disambiguates shared objects.
    task, task_score, task_terms = _match_task(normalized, task_names)
    intent_task, intent_score, intent_terms = _match_intent(normalized)
    if explicit_task and explicit_task != "cleaning":
        task, task_score, task_terms = explicit_task, 100, explicit_terms
    elif intent_task:
        task, task_score, task_terms = intent_task, intent_score, intent_terms

    try:
        parsed = parse_instruction(normalized)
        if task is None:
            task = parsed.task
        else:
            raise ValueError
        return ManualInstruction(
            normalized_instruction=normalized,
            task=acceptance_task(parsed.task),
            object=parsed.object,
            target=_explicit_target(normalized) or parsed.target,
            operation=_infer_operation(normalized, parsed.task, parsed.object),
            used_fallback=False,
            note="按关键词、同义词和任务语义解析成功",
            confidence=100,
            matched_terms=["关键词/同义词"],
        )
    except ValueError:
        pass

    obj, object_score, object_terms = _match_object(normalized)
    if obj and not task:
        candidates, candidate_matches = _top_object_task_candidates(normalized, object_to_task)
        if len(candidates) > 1:
            raise _clarification_error(candidates)
        task = candidates[0]
        obj, object_score, object_terms = candidate_matches[task]
        task_terms = [task_names.get(task, task)]
    elif task and not obj:
        if task in {"maintenance_management", "entertainment_service"}:
            raise ValueError(f"Task {task} requires a specific canonical object")
        obj = DEFAULT_OBJECT_BY_TASK[task]
        object_terms = [f"默认代表物体:{obj}"]
    elif task and obj:
        owner_tasks = object_to_task.get(obj, ())
        if task not in owner_tasks and acceptance_task(task) not in owner_tasks:
            task_obj, task_object_score, task_object_terms = _match_object_for_task(
                normalized, task, object_to_task
            )
            if task_obj:
                obj = task_obj
                object_terms = task_object_terms
                object_score = task_object_score
            elif task_score < object_score + 3:
                task = owner_tasks[0]
                task_terms = [task_names.get(task, task)]

    if not task or not obj:
        catalog = load_catalog()
        object_count = sum(
            len(task_entry.get("objects", []))
            for task_entry in catalog.get("tasks", [])
        )
        raise ValueError(
            f"未能在{len(catalog['acceptance_tasks'])}类任务/"
            f"{object_count}种物体中匹配到合适操作"
        )

    return ManualInstruction(
        normalized_instruction=normalized,
        task=acceptance_task(task),
        object=obj,
        target=_explicit_target(normalized) or _target_for(task, obj, records),
        operation=_infer_operation(normalized, task, obj),
        used_fallback=True,
        note="按关键词、同义词和当前正式清单匹配",
        confidence=task_score + object_score,
        matched_terms=[*task_terms, *object_terms],
    )


def find_expected_actions(resolved: ManualInstruction, records: list[dict]) -> list[str]:
    record = find_benchmark_record(resolved, records)
    return list(record["actions"])


def find_benchmark_record(resolved: ManualInstruction, records: list[dict]) -> dict:
    if not resolved.is_executable:
        raise NonExecutableInstructionError(
            f"Task {resolved.task} with object {resolved.object} is outside the canonical "
            "executable scope; no benchmark action plan is available"
        )
    for split in ("test", "validation", "train"):
        for record in records:
            if (
                record["split"] == split
                and acceptance_task(record["task"]) == resolved.task
                and record["object"] == resolved.object
                and record.get("operation") == resolved.operation
                and record.get("target") == resolved.target
            ):
                return record
    specific_plan = operation_plan(
        resolved.task, resolved.object, resolved.operation, resolved.target
    )
    if specific_plan:
        target, actions = specific_plan
        return {
            "split": "runtime",
            "task": resolved.task,
            "object": resolved.object,
            "operation": resolved.operation,
            "target": target,
            "actions": actions,
            "action_parse": {},
        }
    for split in ("test", "validation", "train"):
        for record in records:
            if (
                record["split"] == split
                and record["task"] == resolved.task
                and record["object"] == resolved.object
            ):
                return record
    cross_intent = CROSS_INTENT_ACTION_PLANS.get((resolved.task, resolved.object))
    if cross_intent:
        return {
            "split": "runtime",
            "task": resolved.task,
            "object": resolved.object,
            "target": cross_intent["target"],
            "actions": list(cross_intent["actions"]),
            "action_parse": {},
        }
    raise ValueError(
        f"No benchmark record found for task={resolved.task}, object={resolved.object}"
    )


def build_manual_result(
    instruction: str,
    resolved: ManualInstruction,
    predicted_actions: list[str],
    expected_actions: list[str],
    *,
    final_actions: list[str] | None = None,
    constraint_applied: bool = False,
    constraint_reason: str | None = None,
    constraint_violations: list[str] | None = None,
    model_used: bool = False,
    prediction_error: str | None = None,
    generation_metadata: dict | None = None,
    checkpoint: str | None = None,
    template_selection: dict | None = None,
) -> dict:
    final_actions = list(final_actions) if final_actions is not None else list(predicted_actions)
    constraint_violations = list(constraint_violations or [])
    scored_actions = predicted_actions if template_selection is not None else final_actions
    completed = (
        model_used
        and prediction_error is None
        and scored_actions == expected_actions
    )
    prediction_available = model_used and prediction_error is None
    records = load_benchmark_records()
    action_parse = find_benchmark_record(resolved, records).get("action_parse", {})
    success_explanation = (
        "模型原始序列与指令意图不一致，系统已采用受约束标准序列"
        if constraint_applied
        else "模型已根据解析后的任务、物体和目标生成动作序列，且与标准动作序列完全一致"
        if model_used
        else "指令已成功匹配，并生成标准动作序列"
    )
    result = {
        "instruction": instruction,
        "resolved": {
            "task": resolved.task,
            "object": resolved.object,
            "target": resolved.target,
            "operation": resolved.operation,
            "used_fallback": resolved.used_fallback,
            "note": resolved.note,
            "confidence": resolved.confidence,
            "matched_terms": resolved.matched_terms,
            "is_executable": resolved.is_executable,
            "is_canonical_object": resolved.is_canonical_object,
        },
        "predicted_actions": predicted_actions,
        "final_actions": final_actions,
        "expected_actions": expected_actions,
        "action_parse": action_parse,
        "prediction_available": prediction_available,
        "model_used": model_used,
        "prediction_error": prediction_error,
        "generation": generation_metadata,
        "checkpoint": checkpoint,
        "template_selection": template_selection,
        "constraint_applied": constraint_applied,
        "constraint_reason": constraint_reason,
        "constraint_violations": constraint_violations,
        "completed": completed,
        "completion_status": "已纠正通过" if completed and constraint_applied else "通过" if completed else "模型生成失败" if prediction_error else "未通过",
        "completion_explanation": (
            success_explanation
            if completed
            else f"指令已成功匹配，但模型生成失败：{prediction_error}"
            if prediction_error
            else "模型输出动作序列与标准动作序列不完全一致"
        ),
    }
    return result


def format_result_text(result: dict) -> str:
    if result.get("result_type") == "multi_device_coordination":
        from tools.multi_device_coordination import format_multi_device_result

        return format_multi_device_result(result)

    if result.get("result_type") == "vla_task_decision":
        from tools.vla_instruction_planner import format_vla_instruction_result

        return format_vla_instruction_result(result)

    resolved = result["resolved"]
    visible_lines = [
        f"你的指令：{result['instruction']}",
        f"我理解为：{describe_resolved_intent(resolved)}。",
        f"建议步骤：{' → '.join(describe_actions(result.get('final_actions') or []))}",
        f"完成情况：{result['completion_status']}",
        f"说明：{result['completion_explanation']}",
    ]
    template_selection = result.get("template_selection")
    if template_selection:
        visible_lines.insert(
            2,
            "预设动作模板：" + " → ".join(template_selection["steps"]),
        )
    if result.get("constraint_applied"):
        visible_lines.append("系统已根据你的指令纠正动作步骤。")

    task = resolved["task"]
    obj = resolved["object"]
    operation = resolved["operation"]
    object_operation_names = {
        ("maintenance_management", "flower_pot", "maintain"): "养护花卉",
        ("entertainment_service", "game_controller", "operate"): "操控游戏手柄",
        ("entertainment_service", "speaker", "operate"): "播放音频",
    }
    operation_display = object_operation_names.get(
        (task, obj, operation), OPERATION_NAMES_ZH.get(operation, operation)
    )
    technical_lines = [
        "技术详情（供检查）：",
        f"匹配任务：{task}",
        f"匹配物体：{obj}",
        f"匹配操作：{operation_display} ({operation})",
        f"目标位置/设备：{resolved['target']}",
        f"匹配说明：{resolved['note']}",
    ]
    if result.get("prediction_available"):
        technical_lines.extend(["", "模型生成动作序列："])
        technical_lines.extend(
            f"{index}. {action}"
            for index, action in enumerate(result["predicted_actions"], start=1)
        )
        generation = result.get("generation") or {}
        if generation:
            technical_lines.append(
                "模型条件："
                f"source={generation.get('source')}，"
                f"verb={generation.get('verb')}，"
                f"score={generation.get('score'):.4f}"
            )
        if result.get("checkpoint"):
            technical_lines.append(f"模型检查点：{result['checkpoint']}")
    if result.get("constraint_applied"):
        technical_lines.extend([
            "",
            "动作约束纠正：",
            str(result.get("constraint_reason") or "模型序列不符合当前指令意图"),
        ])
        technical_lines.extend(
            f"- {violation}" for violation in result.get("constraint_violations") or []
        )
    technical_lines.extend(["", "最终采用动作序列："])
    technical_lines.extend(
        f"{index}. {action}"
        for index, action in enumerate(result.get("final_actions") or [], start=1)
    )
    technical_lines.extend(["", "标准动作序列（仅用于评分）："])
    technical_lines.extend(
        f"{index}. {action}"
        for index, action in enumerate(result["expected_actions"], start=1)
    )
    action_parse = result.get("action_parse") or {}
    if action_parse.get("action_primitives"):
        action_parse = dict(action_parse)
        action_parse["action_primitives"] = [
            PRIMITIVE_NAMES_ZH.get(value, value)
            for value in action_parse["action_primitives"]
        ]
    elif result.get("expected_actions"):
        primitives = []
        for action in result["expected_actions"]:
            primitive = PRIMITIVE_NAMES_ZH.get(action.split("(", 1)[0], action.split("(", 1)[0])
            if primitive not in primitives:
                primitives.append(primitive)
        action_parse = {"action_primitives": primitives}
    if action_parse:
        technical_lines.extend([
            "",
            "动作解析：",
            f"动作原语序列：{' → '.join(action_parse.get('action_primitives', []))}",
            f"视觉观测特征：{action_parse.get('visual_observation', '')}",
            f"执行参数：{action_parse.get('execution_parameters', '')}",
            f"成功判据：{action_parse.get('success_criteria', '')}",
        ])
    return "\n".join([*visible_lines, "", *technical_lines])
