"""Build the review catalogue for all-object coordination scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.manual_instruction_entry import OBJECT_SYNONYMS
from tools.vla_action_templates import load_vla_action_templates
from tools.vla_detailed_action_sequences import load_vla_detailed_action_sequences


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CATALOG = ROOT / "meta" / "compliance_catalog_15x120.json"
BENCHMARK = ROOT / "datasets" / "standard_instruction_action_benchmark.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "019fb5d7-4c5a-7e23-bb89-f7ec997f7068"
    / "all_object_coordination_catalog.json"
)

TASK_NAMES = {
    "cleaning": "清洁",
    "organizing": "整理收纳",
    "smart_cooking": "烹饪加热",
    "appliance_management": "家电管理",
    "security_monitoring": "安防巡检",
    "laundry": "衣物清洗",
    "waste_disposal": "垃圾分类",
    "clothing_care": "衣物护理",
    "window_care": "门窗养护",
    "bedroom_service": "卧室服务",
    "food_serving": "餐饮递送",
    "object_fetching": "物品取送",
    "elderly_assistance": "老人辅助",
    "maintenance_management": "设备维护",
    "entertainment_service": "娱乐服务",
    "indoor_cleaning": "室内卫生清洁",
    "cooking_heating": "烹饪与加热辅助",
    "item_delivery": "物品递送",
    "organizing_storage": "整理收纳",
    "storage_open_close": "储物设施开合",
    "indoor_installation": "室内安装与布置",
    "tableware_placement": "餐具与容器摆放",
    "clothing_entryway": "衣物鞋类与玄关归位",
    "workspace_service": "工作学习区服务",
    "bathroom_personal_care": "卫浴用品与个人卫生服务",
}

DOMAIN_BY_TASK = {
    "cleaning": "全屋清洁", "indoor_cleaning": "全屋清洁",
    "organizing": "整理收纳", "organizing_storage": "整理收纳",
    "smart_cooking": "厨房与用餐", "cooking_heating": "厨房与用餐",
    "tableware_placement": "厨房与用餐", "food_serving": "厨房与用餐",
    "storage_open_close": "厨房与用餐", "item_delivery": "生活补给",
    "object_fetching": "生活补给", "laundry": "衣物与玄关",
    "clothing_care": "衣物与玄关", "clothing_entryway": "衣物与玄关",
    "window_care": "门窗与环境", "appliance_management": "门窗与环境",
    "maintenance_management": "维护与节能", "security_monitoring": "安防与应急",
    "bedroom_service": "卧室与就寝", "elderly_assistance": "健康辅助",
    "entertainment_service": "娱乐休闲", "workspace_service": "工作学习",
    "indoor_installation": "安装布置", "bathroom_personal_care": "卫浴与个人护理",
    "waste_disposal": "垃圾与回收",
}

SCENE_PURPOSE = {
    "全屋清洁": "完成分区清洁并检查工具与表面状态",
    "整理收纳": "完成物品分类、归位与稳定性复核",
    "厨房与用餐": "准备烹饪和用餐环境并安全收尾",
    "生活补给": "完成生活物品取送、摆放与交接",
    "衣物与玄关": "完成衣物鞋类处理、分类和归位",
    "门窗与环境": "调整室内环境并检查门窗设备状态",
    "维护与节能": "执行设备检查维护并恢复安全状态",
    "安防与应急": "完成住宅安全巡检和应急物品确认",
    "卧室与就寝": "整理卧室并准备舒适就寝环境",
    "健康辅助": "准备健康辅助用品并检查可用状态",
    "娱乐休闲": "布置休闲环境并启动所需设备",
    "工作学习": "整理工作学习区并布置办公用品",
    "安装布置": "完成室内物品安装、摆位与稳固检查",
    "卫浴与个人护理": "清洁卫浴区域并归位个人用品",
    "垃圾与回收": "完成垃圾识别、分类投放与区域清洁",
}

COORDINATION_CLUSTERS = (
    ("清洁卫浴与回收", ("全屋清洁", "卫浴与个人护理", "垃圾与回收")),
    ("厨房收纳与补给", ("厨房与用餐", "整理收纳", "生活补给")),
    ("衣物卧室与环境", ("衣物与玄关", "卧室与就寝", "门窗与环境")),
    ("安全健康与维护", ("安防与应急", "健康辅助", "维护与节能")),
    ("工作娱乐与布置", ("工作学习", "娱乐休闲", "安装布置")),
)

CLUSTER_PURPOSE = {
    "清洁卫浴与回收": "完成清洁、卫浴用品整理和垃圾分类收尾",
    "厨房收纳与补给": "完成备餐、餐具摆放、物品补给和厨房归位",
    "衣物卧室与环境": "完成衣物处理、卧室整理和室内环境调节",
    "安全健康与维护": "完成安全巡检、健康辅助用品准备和设备维护",
    "工作娱乐与布置": "完成工作区布置、设备准备和休闲物品安排",
}

INTENT_CONFIG = {
    "floor_cleaning": ("清洁卫浴与回收", ("帮我把地面清理干净", "地面该清洁了", "帮我清洁一下地面")),
    "bathroom_care": ("清洁卫浴与回收", ("帮我整理洗漱用品", "帮我整理一下浴室", "帮我清洁卫浴设施", "帮我做好洗漱准备")),
    "dish_cleanup": ("清洁卫浴与回收", ("帮我把餐具洗干净", "饭后帮我收拾餐具", "帮我清理一下厨具")),
    "cleaning_tools": ("清洁卫浴与回收", ("帮我准备清洁工具", "帮我整理清洁用品")),
    "waste_recycling": ("清洁卫浴与回收", ("帮我把垃圾分类处理", "垃圾该收一下了", "帮我处理可回收物")),
    "cooking": ("厨房收纳与补给", ("我要开始做饭了", "帮我准备一顿饭", "帮我准备早餐", "帮我准备午餐", "帮我准备晚餐", "我要烤点东西", "帮我备好锅具", "准备好烹饪环境")),
    "dining": ("厨房收纳与补给", ("帮我把餐桌布置好", "帮我准备餐具", "帮我摆好饮品和餐具")),
    "kitchen_tools": ("厨房收纳与补给", ("帮我整理厨房用具", "帮我备好烹饪用具")),
    "cold_storage": ("厨房收纳与补给", ("帮我收拾冰箱",)),
    "appliance_storage": ("厨房收纳与补给", ("帮我检查厨房电器门架",)),
    "kitchen_cabinets": ("厨房收纳与补给", ("帮我整理厨房柜子",)),
    "refreshments": ("厨房收纳与补给", ("帮我准备饮料", "帮我准备下午茶")),
    "home_organizing": ("厨房收纳与补给", ("帮我把物品收纳好", "家里该整理一下了", "帮我整理柜子和收纳盒", "帮我收拾一下客厅")),
    "clothing_care": ("衣物卧室与环境", ("我要洗衣服了", "帮我护理这批衣物", "帮我整理换季衣物", "帮我把衣物处理好", "帮我准备洗衣用品", "帮我收好干净衣物")),
    "entryway": ("衣物卧室与环境", ("帮我整理一下玄关", "帮我整理鞋帽", "我要准备出门了")),
    "bedtime_devices": ("衣物卧室与环境", ("我要睡觉了",)),
    "bedroom_tidy": ("衣物卧室与环境", ("帮我整理一下卧室", "准备好就寝环境", "帮我做好睡前整理")),
    "window_care": ("衣物卧室与环境", ("帮我清理一下门窗", "帮我检查窗户", "窗户该保养了")),
    "home_environment": ("衣物卧室与环境", ("帮我调好室内环境", "家里有点闷了")),
    "going_out": ("衣物卧室与环境", ("我要出门了", "帮我准备随身物品")),
    "safety": ("安全健康与维护", ("帮我检查一下家里安全", "我要做一次安全巡检", "帮我检查门锁和消防")),
    "health": ("安全健康与维护", ("我要做健康检查了", "帮我准备护理用品", "帮我准备急救用品")),
    "balcony_care": ("安全健康与维护", ("帮我照料一下阳台",)),
    "workspace": ("工作娱乐与布置", ("我要开始办公了", "帮我整理一下书桌", "准备好学习环境", "帮我准备办公用品", "我要开始学习了", "帮我收拾工作区")),
    "entertainment": ("工作娱乐与布置", ("我要休息娱乐一下",)),
    "installation": ("工作娱乐与布置", ("帮我布置一下房间", "帮我架好拍摄设备")),
}

FLOOR_OBJECT_IDS = {
    "bathroom_floor", "carpet", "laminate_floor", "marble_floor",
    "tile_floor", "vinyl_floor", "wood_floor", "robot_vacuum",
}
DISH_OBJECT_IDS = {"mug", "bowl", "cup", "fork", "knife", "plate", "pot", "spoon"}
BEDTIME_DEVICE_IDS = {"air_conditioner", "electric_curtain", "bedroom_lamp"}
BEDROOM_OBJECT_IDS = {
    "bed", "clothes_hanger", "laundry_basket", "mattress", "nightstand", "pillow", "quilt",
}


def _intent_for(item: dict[str, Any]) -> str:
    object_id, task = item["object_id"], item["task"]
    if object_id in FLOOR_OBJECT_IDS:
        return "floor_cleaning"
    if object_id in DISH_OBJECT_IDS:
        return "dish_cleanup"
    if object_id in BEDTIME_DEVICE_IDS:
        return "bedtime_devices"
    if object_id in BEDROOM_OBJECT_IDS:
        return "bedroom_tidy"
    if object_id in {"bathtub", "faucet", "mirror", "shower", "sink", "soap", "toilet"} or task == "bathroom_personal_care":
        return "bathroom_care"
    if object_id in {"vla_002", "vla_003", "vla_004"}:
        return "cleaning_tools"
    if task in {"waste_disposal", "indoor_cleaning"}:
        return "waste_recycling"
    if task in {"smart_cooking", "cooking_heating"}:
        return "cooking"
    if task == "food_serving" or object_id in {"vla_050", "vla_054", "vla_058"}:
        return "dining"
    if task == "tableware_placement":
        return "kitchen_tools"
    if object_id == "refrigerator" or object_id in {"vla_035", "vla_036", "vla_037"}:
        return "cold_storage"
    if object_id in {"vla_038", "vla_041", "vla_042", "vla_043", "vla_044"}:
        return "appliance_storage"
    if task == "storage_open_close":
        return "kitchen_cabinets"
    if task == "item_delivery" and object_id != "vla_023":
        return "refreshments"
    if task in {"organizing", "organizing_storage"}:
        return "home_organizing"
    if task in {"laundry", "clothing_care"} or object_id == "washing_machine":
        return "clothing_care"
    if object_id in {"vla_061", "vla_062"}:
        return "clothing_care"
    if task == "clothing_entryway":
        return "entryway"
    if task == "bedroom_service":
        return "bedroom_tidy"
    if task == "window_care" and object_id == "balcony_railing":
        return "balcony_care"
    if task == "window_care":
        return "window_care"
    if task == "appliance_management" and object_id in {"television", "remote_control"}:
        return "entertainment"
    if task == "appliance_management" and object_id == "reading_lamp":
        return "workspace"
    if task == "appliance_management":
        return "home_environment"
    if task == "object_fetching" and object_id in {"book", "laptop"} or object_id == "vla_023":
        return "workspace"
    if task == "object_fetching":
        return "going_out"
    if task == "security_monitoring":
        return "safety"
    if task == "elderly_assistance":
        return "health"
    if task == "maintenance_management":
        return "balcony_care"
    if task == "workspace_service":
        return "workspace"
    if task == "entertainment_service":
        return "entertainment"
    if task == "indoor_installation":
        return "installation"
    raise ValueError(f"未定义用途：{object_id}/{task}")


def _zh_name(object_id: str) -> str:
    aliases = OBJECT_SYNONYMS.get(object_id, [])
    return next((item for item in aliases if any("\u4e00" <= ch <= "\u9fff" for ch in item)), object_id)


def _original_objects() -> list[dict[str, Any]]:
    catalogue = json.loads(ORIGINAL_CATALOG.read_text(encoding="utf-8"))
    records = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    memberships: dict[str, list[str]] = defaultdict(list)
    for task in catalogue["tasks"]:
        for object_id in task["objects"]:
            memberships[object_id].append(task["id"])
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_object[record["object"]].append(record)
    result = []
    for object_id in sorted(memberships):
        record = by_object[object_id][0]
        result.append({
            "source": "原有目录",
            "object_id": object_id,
            "object_name_zh": _zh_name(object_id),
            "task": record["task"],
            "all_tasks": memberships[object_id],
            "operation": record.get("operation") or "canonical",
            "target": record.get("target") or "none",
            "instruction": record.get("instruction") or f"操作{_zh_name(object_id)}",
            "original_actions": list(record["actions"]),
            "domain": DOMAIN_BY_TASK.get(record["task"], "综合生活服务"),
        })
    return result


def _vla_objects() -> list[dict[str, Any]]:
    templates = {row["object_id"]: row for row in load_vla_action_templates()["candidates"]}
    details = load_vla_detailed_action_sequences()["sequences"]
    result = []
    for detail in details:
        template = templates[detail["object_id"]]
        result.append({
            "source": "VLA",
            "object_id": detail["object_id"],
            "object_name_zh": detail["object_name_zh"],
            "task": detail["task_id"],
            "all_tasks": [detail["task_id"]],
            "operation": detail["source_operation_label"],
            "target": "按VLA操作标签确定",
            "instruction": f"执行VLA任务：{detail['source_operation_label']}（对象：{detail['object_name_zh']}）",
            "original_actions": list(template["steps"]),
            "approved_detailed_actions": list(detail["steps"]),
            "domain": DOMAIN_BY_TASK[detail["task_id"]],
        })
    return result


def _detailed_actions(item: dict[str, Any]) -> list[str]:
    if item["source"] == "VLA":
        return list(item["approved_detailed_actions"])
    name = item["object_name_zh"]
    task = item["task"]
    if task == "cleaning":
        return [f"识别{name}的材质、表面状态和污染类型", "根据材质选择清洁工具、清洁剂和干湿方式", "先处理可见杂物并规划由内向外的全覆盖路径", f"以不损伤{name}的力度分区清洁并覆盖边角", "复查污渍、水痕、残留物和周边安全状态"]
    if task in {"smart_cooking", "food_serving"}:
        return [f"检查{name}洁净度、容量、耐热性和当前状态", "确认关联食材、器具及目标位置均已准备", "按先准备后加热或摆放的依赖顺序执行", "控制温度、时间、份量或摆放间距", "完成后检查防烫、断电、稳定性和交叉污染"]
    if task in {"organizing", "bedroom_service", "object_fetching"}:
        return [f"识别{name}当前位置、类别、数量和目标区域", "清理搬运路径并确认抓取点与承重", "按类别、使用频率和尺寸确定归位顺序", "保持正面朝向、合理间距和重心稳定", "复核无遮挡、无挤压且便于再次取用"]
    if task in {"laundry", "clothing_care"}:
        return [f"读取{name}材质、颜色、污渍和护理标签", "按材质与颜色分类并选择清洗或护理方式", "设置水温、转速、力度或熨烫温度", "执行清洗、晾干、折叠或熨烫并避免变形", "检查干燥度、平整度和归位状态"]
    if task in {"appliance_management", "maintenance_management", "window_care"}:
        return [f"定位{name}并读取电源、开合或运行状态", "检查周围障碍、供电和安全联锁条件", "执行目标开关、开合、清洁或维护动作", "设置必要参数并监测响应及异常提示", "确认最终状态与同设备互斥目标不存在冲突"]
    if task in {"security_monitoring", "elderly_assistance"}:
        return [f"检查{name}外观、供电、有效期或校准状态", "确认使用对象、环境条件和安全前置要求", "按规范完成检查、测量或辅助操作", "读取状态或数值并与安全阈值比较", "异常时停止后续动作并提醒、记录或上报"]
    if task in {"waste_disposal"}:
        return [f"识别{name}材质、污染程度和可回收属性", "选择对应垃圾类别、容器和防护方式", "密封或整理后沿安全路径搬运", "投入正确区域并避免液体泄漏或锐物暴露", "清洁接触区域并复核分类正确性"]
    return [f"识别{name}位置、姿态、可抓取区域和目标位置", "检查搬运路径、障碍物及关联对象状态", "选择稳定抓取点并控制夹持力度", "按依赖顺序移动、放置、连接或固定", "释放支撑后复核位置、朝向和稳定性"]


def _phase_for(task: str) -> tuple[str, str]:
    if task in {"security_monitoring", "maintenance_management"}:
        return "阶段1：状态与安全检查", "条件"
    if task in {"cleaning", "organizing", "laundry", "object_fetching"}:
        return "阶段2：并行准备", "并行"
    if task in {"smart_cooking", "clothing_care", "elderly_assistance"}:
        return "阶段3：核心操作", "顺序"
    return "阶段4：归位与确认", "顺序"


def _scenario_groups(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        intent_key = _intent_for(item)
        item["intent_key"] = intent_key
        item["coordination_cluster"] = INTENT_CONFIG[intent_key][0]
        by_intent[intent_key].append(item)
    groups: list[list[dict[str, Any]]] = []
    for intent_key in INTENT_CONFIG:
        ordered = sorted(by_intent[intent_key], key=lambda x: (x["task"], x["source"], x["object_id"]))
        quotient, remainder = divmod(len(ordered), 5)
        sizes = [5] * quotient
        if remainder == 1:
            if not sizes:
                sizes = [1]
            else:
                sizes[-1] = 4
                sizes.append(2)
        elif remainder == 2:
            sizes.append(2)
        elif remainder == 3:
            sizes.append(3)
        elif remainder == 4:
            sizes.append(4)
        cluster_groups, offset = [], 0
        for size in sizes:
            cluster_groups.append(ordered[offset:offset + size])
            offset += size
        if any(len(group) < 2 for group in cluster_groups):
            raise ValueError(f"用途 {intent_key} 的对象不足以形成联动场景")
        if len(cluster_groups) > len(INTENT_CONFIG[intent_key][1]):
            raise ValueError(f"用途 {intent_key} 缺少足够的口语命令")
        groups.extend(cluster_groups)
    if sum(map(len, groups)) != len(items):
        raise ValueError("场景分组未覆盖全部对象")
    return groups


def build_catalog() -> dict[str, Any]:
    original = _original_objects()
    vla = _vla_objects()
    if (len(original), len(vla)) != (123, 82):
        raise ValueError(f"对象数量不符合审核范围：原有{len(original)}，VLA{len(vla)}")
    groups = _scenario_groups([*original, *vla])
    scenarios, object_details = [], []
    intent_command_indexes: dict[str, int] = defaultdict(int)
    for index, group in enumerate(groups, start=1):
        scenario_id = f"MDC-{index:03d}"
        primary_domain = max({item["domain"] for item in group}, key=lambda d: sum(x["domain"] == d for x in group))
        cluster_name = group[0]["coordination_cluster"]
        intent_key = group[0]["intent_key"]
        if any(item["intent_key"] != intent_key for item in group):
            raise ValueError(f"场景 {scenario_id} 混入不同用途对象")
        tasks = sorted({item["task"] for item in group})
        command_index = intent_command_indexes[intent_key]
        command = INTENT_CONFIG[intent_key][1][command_index]
        intent_command_indexes[intent_key] += 1
        phases = "阶段1状态检查；阶段2并行准备；阶段3顺序核心操作；阶段4归位确认；阶段5冲突复核"
        scenario = {
            "scenario_id": scenario_id, "category": cluster_name,
            "name": f"{cluster_name}联动{index:02d}", "command": command,
            "intent_key": intent_key,
            "task_count": len(tasks), "object_count": len(group),
            "tasks": tasks,
            "original_objects": [f"{x['object_name_zh']}({x['object_id']})" for x in group if x["source"] == "原有目录"],
            "vla_objects": [f"{x['object_name_zh']}({x['object_id']})" for x in group if x["source"] == "VLA"],
            "execution_phases": phases, "execution_modes": "条件检查 + 并行准备 + 顺序执行",
            "trigger_condition": "用户下发场景命令；相关对象可用且安全前置条件满足",
            "conflict_rules": "互斥开关/开合不得并存；同一对象不得有两个目标位置；湿式清洁不得与带电维护并行",
            "expected_output": "识别全部子任务与对象，输出原动作和额外细分动作，形成依赖明确、无冲突的分阶段方案",
            "qualification": "全部对象解析成功；动作非空；依赖正确；冲突检查通过；完成情况为通过",
            "review_status": "待审核", "review_comment": "",
        }
        scenarios.append(scenario)
        for item in group:
            phase, mode = _phase_for(item["task"])
            detailed = _detailed_actions(item)
            original_actions = item["original_actions"]
            original_text = " → ".join(original_actions)
            detailed_text = " → ".join(detailed)
            condition = f"{primary_domain}场景；{TASK_NAMES.get(item['task'], item['task'])}任务；对象为{item['object_name_zh']}；相关设备、工具和路径处于安全可用状态。"
            test_steps = f"1. 下发联动命令“{command}”；\n2. 调用“{item['instruction']}”子任务并识别任务、物体和操作；\n3. 生成并输出该物体的额外细分动作序列；\n4. 检查执行阶段、依赖、目标状态和冲突规则；\n5. 记录结果并等待审核。"
            expected = f"识别任务：{item['task']}；物体：{item['object_name_zh']}（{item['object_id']}）；操作：{item['operation']}；原动作序列：{original_text}；额外细分动作序列：{detailed_text}；执行阶段：{phase}；目标状态：操作完成且状态可复核；满足全部判定条件时完成情况：通过。"
            object_details.append({
                "scenario_id": scenario_id, "category": cluster_name,
                "intent_key": intent_key,
                "test_condition": condition, "test_steps": test_steps, "expected_output": expected,
                "source": item["source"], "task": item["task"], "task_name_zh": TASK_NAMES.get(item["task"], item["task"]),
                "object_id": item["object_id"], "object_name_zh": item["object_name_zh"],
                "subtask_instruction": item["instruction"], "operation": item["operation"],
                "original_actions": original_actions, "detailed_actions": detailed,
                "prerequisites": "对象可定位；目标区域可达；关联工具/设备状态可读取；安全联锁有效",
                "control_parameters": "依据材质/设备说明设置力度、速度、温度、时间、间距或安全阈值",
                "execution_phase": phase, "execution_mode": mode,
                "dependencies": "同场景阶段1状态检查完成；显式关联对象先满足前置状态",
                "target_state": "操作完成、对象位置/设备状态明确且可复核",
                "conflict_rule": "不得与同对象互斥动作并存；不得绕过前置依赖；异常时停止并记录",
                "qualification": "任务、物体和操作识别正确；两套动作序列独立完整；依赖和目标状态明确；冲突检查通过",
                "review_status": "待审核", "review_comment": "",
            })
    coverage = [{
        "source": row["source"], "task": row["task"], "object_id": row["object_id"],
        "object_name_zh": row["object_name_zh"], "scenario_ids": [row["scenario_id"]],
        "detail_count": 1, "covered": True, "review_status": "待审核", "review_comment": "",
    } for row in object_details]
    catalog = {
        "format": "all_object_coordination_review_v1",
        "summary": {"original_unique_objects": len(original), "vla_objects": len(vla), "total_objects": len(object_details), "scenario_count": len(scenarios)},
        "scenarios": scenarios, "object_details": object_details, "coverage": coverage,
    }
    if not all(2 <= row["object_count"] <= 5 for row in scenarios):
        raise ValueError("存在不满足多物体联动规模要求的场景")
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalog = build_catalog()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    s = catalog["summary"]
    print(f"scenarios={s['scenario_count']}; original={s['original_unique_objects']}/123; VLA={s['vla_objects']}/82; total={s['total_objects']}/205")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
