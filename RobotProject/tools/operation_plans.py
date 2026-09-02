"""Operation-specific action plans shared by parsing and benchmark generation."""

from __future__ import annotations


DEFAULT_OPERATION_BY_TASK = {
    "floor_cleaning": "clean",
    "organizing": "organize",
    "smart_cooking": "cook",
    "appliance_management": "turn_on",
    "security_monitoring": "inspect",
    "laundry": "wash",
    "dish_washing": "wash",
    "waste_disposal": "dispose",
    "clothing_care": "organize",
    "window_care": "clean",
    "bathroom_cleaning": "clean",
    "bedroom_service": "organize",
    "food_serving": "serve",
    "object_fetching": "fetch",
    "elderly_assistance": "assist",
    "maintenance_management": "maintain",
    "entertainment_service": "operate",
}


def canonical_acceptance_action_plan(
    task: str, obj: str, operation: str | None = None
) -> tuple[str, list[str]]:
    """Return the single canonical plan for redesigned acceptance tasks."""
    operation = operation or DEFAULT_OPERATION_BY_TASK[task]
    if task == "maintenance_management":
        if obj == "flower_pot" and operation == "water_plants":
            return "flower_pot", [
                "locate(watering_can)",
                "grasp(watering_can)",
                "fill(watering_can)",
                "move(flower_pot)",
                "water(flower_pot,watering_can)",
                "place(watering_can)",
            ]
        if obj == "flower_pot" and operation in {"maintain", "care_plant"}:
            return "flower_pot", [
                "locate(flower_pot)",
                "inspect_plant_and_pests(flower_pot)",
                "remove_pests(flower_pot)",
                "organize(flower_pot)",
                "inspect(flower_pot)",
            ]
        if operation != "maintain":
            raise ValueError(f"unsupported maintenance operation: {operation}")
        custom_plans = {
            "flower_pot": [
                "locate(flower_pot)",
                "inspect_plant_and_pests(flower_pot)",
                "remove_pests(flower_pot)",
                "organize(flower_pot)",
                "inspect(flower_pot)",
            ],
            "watering_can": [
                "locate(watering_can)",
                "grasp(watering_can)",
                "fill(watering_can)",
                "move(flower_pot)",
                "water(flower_pot,watering_can)",
                "place(watering_can)",
            ],
        }
        if obj in custom_plans:
            return "flower_pot", custom_plans[obj]
        specialized = {
            "robot_vacuum": "service_brush",
            "fire_alarm": "test_alarm",
            "washing_machine": "clean_drum",
            "air_conditioner": "clean_filter",
            "refrigerator": "check_seal",
            "fan": "clean_blades",
            "electric_curtain": "service_track",
            "smart_lock": "test_lock",
        }
        service = specialized.get(obj)
        if service is None:
            raise ValueError(f"unsupported maintenance object: {obj}")
        return "none", [
            f"locate({obj})",
            f"inspect({obj})",
            f"{service}({obj})",
            f"maintain({obj})",
            f"confirm_state({obj})",
        ]

    if task != "entertainment_service":
        raise ValueError(f"not a redesigned acceptance task: {task}")
    if obj == "television":
        if operation == "turn_off":
            return "none", [
                "locate(television)", "turn_off(television)", "confirm_state(television)"
            ]
        if operation == "turn_on":
            return "none", [
                "locate(television)", "turn_on(television)", "confirm_state(television)"
            ]
        if operation == "operate":
            return "none", [
                "locate(television)",
                "turn_on(television)",
                "play(television)",
                "confirm_state(television)",
            ]
    if obj == "game_controller" and operation in {"operate", "operate_game_controller"}:
        return "entertainment_device", [
            "locate(game_controller)",
            "grasp(game_controller)",
            "connect(game_controller,entertainment_device)",
            "start_game(entertainment_device)",
            "control(game_controller,entertainment_device)",
            "place(game_controller)",
        ]
    if obj == "speaker" and operation in {"operate", "play_audio"}:
        return "audio_source", [
            "locate(speaker)",
            "turn_on(speaker)",
            "connect(speaker,audio_source)",
            "play(speaker)",
            "adjust_volume(speaker)",
        ]
    if operation != "operate":
        raise ValueError(f"unsupported entertainment operation: {obj}/{operation}")
    if obj == "remote_control":
        return "television", [
            "locate(remote_control)",
            "grasp(remote_control)",
            "aim(remote_control,television)",
            "press(remote_control)",
            "control(remote_control,television)",
            "confirm_state(television)",
        ]
    if obj in {"laptop", "mobile_phone"}:
        return "none", [
            f"locate({obj})",
            f"turn_on({obj})",
            f"play({obj})",
            f"control({obj})",
            f"confirm_state({obj})",
        ]
    if obj == "book":
        return "entertainment_area", [
            "locate(book)", "grasp(book)", "move(entertainment_area)", "open(book)", "present(book)"
        ]
    if obj == "toy":
        return "entertainment_area", [
            "locate(toy)", "grasp(toy)", "move(entertainment_area)", "play(toy)", "confirm_state(toy)"
        ]
    raise ValueError(f"unsupported entertainment object: {obj}")


def operation_plan(
    task: str, obj: str, operation: str, target: str | None = None
) -> tuple[str, list[str]] | None:
    """Return a safe action plan when an operation needs task-specific behavior."""
    if task == "cleaning" and operation == "wash" and target == "sink":
        return "sink", [
            f"locate({obj})",
            f"grasp({obj})",
            "move(sink)",
            f"wash({obj})",
            f"rinse({obj})",
            f"place({obj})",
        ]

    if task in {"maintenance_management", "entertainment_service"}:
        return canonical_acceptance_action_plan(task, obj, operation)
    if task == "window_care":
        plans = {
            "open": [f"locate({obj})", f"open({obj})", f"inspect({obj})"],
            "close": [f"locate({obj})", f"close({obj})", f"inspect({obj})"],
            "clean": [f"locate({obj})", f"wipe({obj})", f"inspect({obj})"],
            "inspect": [f"locate({obj})", f"inspect({obj})"],
        }
        actions = plans.get(operation)
        return ("none", actions) if actions else None

    if task == "appliance_management":
        plans = {
            "turn_on": [f"locate({obj})", f"turn_on({obj})", f"inspect({obj})"],
            "turn_off": [f"locate({obj})", f"turn_off({obj})", f"inspect({obj})"],
            "open": [f"locate({obj})", f"open({obj})", f"inspect({obj})"],
            "close": [f"locate({obj})", f"close({obj})", f"inspect({obj})"],
            "inspect": [f"locate({obj})", f"inspect({obj})"],
        }
        actions = plans.get(operation)
        return ("none", actions) if actions else None

    if task == "floor_cleaning" and obj == "robot_vacuum" and operation == "robot_clean":
        return "floor", [
            "locate(robot_vacuum)",
            "turn_on(robot_vacuum)",
            "clean(robot_vacuum)",
            "inspect(robot_vacuum)",
            "turn_off(robot_vacuum)",
        ]

    if task == "security_monitoring" and operation == "lock":
        return "none", [f"locate({obj})", f"lock({obj})", f"inspect({obj})"]

    if task == "security_monitoring" and operation == "press":
        return "none", [f"locate({obj})", f"press({obj})", f"inspect({obj})"]

    if task == "bedroom_service" and obj == "bedroom_lamp":
        plans = {
            "turn_on": ["locate(bedroom_lamp)", "turn_on(bedroom_lamp)", "inspect(bedroom_lamp)"],
            "turn_off": ["locate(bedroom_lamp)", "turn_off(bedroom_lamp)", "inspect(bedroom_lamp)"],
        }
        actions = plans.get(operation)
        return ("bedroom", actions) if actions else None

    if task == "clothing_care":
        if operation == "inspect":
            return "none", [f"locate({obj})", f"inspect({obj})"]
        if operation == "iron":
            return "none", [
                f"locate({obj})",
                "grasp(iron)",
                f"iron({obj})",
                f"inspect({obj})",
            ]
        if operation in {"organize", "fold"}:
            return "wardrobe", [
                f"locate({obj})",
                f"grasp({obj})",
                f"fold({obj})",
                "move(wardrobe)",
                f"place({obj})",
            ]

    if task == "elderly_assistance" and operation == "inspect":
        return "none", [f"locate({obj})", f"inspect({obj})"]

    if task == "laundry" and operation == "wash":
        return "washer", [
            f"locate({obj})",
            f"grasp({obj})",
            "move(washer)",
            f"wash({obj})",
            f"dry({obj})",
            f"inspect({obj})",
        ]

    if task == "food_serving" and operation == "serve" and target == "table":
        return "table", [
            f"locate({obj})",
            f"grasp({obj})",
            "move(table)",
            f"place({obj})",
        ]

    if task == "food_serving" and obj == "water_cup" and operation == "pour_water":
        return "person", [
            "locate(water_cup)",
            "grasp(water_cup)",
            "move(water_source)",
            "fill(water_cup)",
            "pour(water_cup)",
            "move(person)",
            "place(water_cup)",
        ]

    if task == "object_fetching" and obj == "water_cup" and operation == "fetch":
        return "person", [
            "locate(water_cup)",
            "grasp(water_cup)",
            "move(person)",
            "release(water_cup)",
        ]
    return None


def acceptance_reference_action_plan(task: str, obj: str) -> list[str]:
    """Compatibility wrapper; all consumers should use the canonical function."""
    return canonical_acceptance_action_plan(task, obj)[1]
