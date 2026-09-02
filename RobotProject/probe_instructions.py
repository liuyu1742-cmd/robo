from tools.manual_instruction_entry import load_benchmark_records, resolve_manual_instruction

INSTRUCTIONS = [
    "把书桌上的文件和文具整理整齐。",
    "睡前请关闭阅读灯并确认已经熄灭。",
    "出门前检查智能门锁是否已经锁好。",
    "请把茶几上的遥控器递给我。",
    "请清洁马桶内外表面。",
    "把门口的废电池放入专用回收盒。",
    "烹饪前请启动电饭煲煮饭。",
    "把茶几上的玩具放进收纳箱。",
    "检查玄关的可视门铃是否正常。",
    "把药盒从床头柜拿给我。",
    "把门口的拐杖拿给我。",
    "清理洗手台并把肥皂放回皂盒。",
    "把厨房的玻璃瓶按可回收物分类处理。",
    "晚饭后把灶上的平底锅收好。",
    "请打开空调并设为舒适温度。",
    "把衣柜里的衣物重新整理好。",
]

records = load_benchmark_records()
for instruction in INSTRUCTIONS:
    try:
        r = resolve_manual_instruction(instruction, benchmark_records=records)
        print(instruction)
        print(r.task, r.object, r.operation, r.target)
    except Exception as exc:
        print(instruction)
        print(type(exc).__name__, exc)
