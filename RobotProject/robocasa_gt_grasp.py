"""Run a ground-truth-pose pick-and-place smoke test in RoboCasa."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np


ROOT = Path(__file__).resolve().parent
THIRD_PARTY = ROOT.parent / "third_party"
sys.path.insert(0, str(THIRD_PARTY / "robosuite"))
sys.path.insert(0, str(THIRD_PARTY / "robocasa"))

import robocasa  # noqa: E402,F401 - registers RoboCasa Gym environments
from robocasa.utils import object_utils as OU  # noqa: E402
from robocasa.environments.kitchen.atomic.kitchen_pick_place import (  # noqa: E402
    PickPlaceCounterToCabinet,
    PickPlaceCounterToSink,
)
from robosuite.utils.control_utils import orientation_error  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-group", default="mustard")
    parser.add_argument("--max-move-steps", type=int, default=100)
    parser.add_argument("--grasp-height-offset", type=float, default=0.08)
    parser.add_argument("--grasp-mode", choices=("top", "side"), default="top")
    parser.add_argument("--side-clearance", type=float, default=0.035)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--cabinet-depth-fraction", type=float, default=0.50)
    parser.add_argument("--destination", choices=("cabinet", "sink"), default="cabinet")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "robocasa_gt_grasp",
    )
    return parser.parse_args()


def action(
    position: np.ndarray,
    gripper_closed: bool,
    base_motion: np.ndarray | None = None,
    base_mode: bool = False,
    rotation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    return {
        "action.end_effector_position": np.asarray(position, dtype=np.float32),
        "action.end_effector_rotation": np.asarray(
            np.zeros(3) if rotation is None else rotation, dtype=np.float32
        ),
        "action.gripper_close": np.array([float(gripper_closed)], dtype=np.float32),
        "action.base_motion": np.asarray(
            np.zeros(4) if base_motion is None else base_motion, dtype=np.float32
        ),
        "action.control_mode": np.array([float(base_mode)], dtype=np.float32),
    }


def camera_frame(observation: dict) -> np.ndarray:
    return np.asarray(observation["video.robot0_agentview_left"])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.object_scale <= 0:
        raise ValueError("--object-scale must be positive")
    environment_class = (
        PickPlaceCounterToCabinet
        if args.destination == "cabinet"
        else PickPlaceCounterToSink
    )
    if args.object_scale != 1.0:
        original_get_obj_cfgs = environment_class._get_obj_cfgs

        def scaled_get_obj_cfgs(environment):
            configs = original_get_obj_cfgs(environment)
            for config in configs:
                if config.get("name") == "obj":
                    config["object_scale"] = args.object_scale
            return configs

        environment_class._get_obj_cfgs = scaled_get_obj_cfgs

    env = gym.make(
        f"robocasa/PickPlaceCounterTo{args.destination.title()}",
        split="pretrain",
        seed=args.seed,
        obj_registries=("lightwheel",),
        obj_groups=args.object_group,
        disable_env_checker=True,
    )
    observation, _ = env.reset(seed=args.seed)
    raw = env.unwrapped.env
    robot = raw.robots[0]
    controller = robot.composite_controller.part_controllers["right"]
    eef_site_id = robot.eef_site_id["right"]
    target_body_id = raw.obj_body_id["obj"]

    video_path = args.output / "gt_pick_place.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (256, 256),
    )
    phase_log: list[dict] = []
    success = False

    def record() -> None:
        frame = camera_frame(observation)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def eef_position() -> np.ndarray:
        return np.array(raw.sim.data.site_xpos[eef_site_id])

    def object_position() -> np.ndarray:
        return np.array(raw.sim.data.body_xpos[target_body_id])

    def eef_orientation() -> np.ndarray:
        return np.array(raw.sim.data.site_xmat[eef_site_id]).reshape(3, 3)

    def move_to_pose(
        name: str,
        target: np.ndarray,
        target_orientation: np.ndarray,
        closed: bool,
        position_tolerance: float = 0.025,
        orientation_tolerance: float = 0.08,
    ) -> bool:
        nonlocal observation
        reached = False
        orientation_distance = float("inf")
        for step in range(args.max_move_steps * 3):
            current = eef_position()
            error_base = controller.world_to_origin_frame(target) - controller.world_to_origin_frame(current)
            desired_origin = controller.origin_ori.T @ target_orientation
            current_origin = controller.origin_ori.T @ eef_orientation()
            rotation_error = orientation_error(desired_origin, current_origin)
            distance = float(np.linalg.norm(error_base))
            orientation_distance = float(np.linalg.norm(rotation_error))
            if distance <= position_tolerance and orientation_distance <= orientation_tolerance:
                reached = True
                break
            position_command = np.clip(error_base / 0.05, -0.7, 0.7)
            rotation_command = np.clip(rotation_error / 0.5, -1.0, 1.0)
            observation, _, _, _, _ = env.step(
                action(position_command, closed, rotation=rotation_command)
            )
            record()
        phase_log.append(
            {
                "phase": name,
                "reached": reached,
                "steps": step + 1,
                "target_world": target.tolist(),
                "eef_world": eef_position().tolist(),
                "orientation_error": orientation_distance,
                "target_orientation": target_orientation.tolist(),
                "eef_orientation": eef_orientation().tolist(),
            }
        )
        print(f"{name}: {'reached' if reached else 'timeout'}")
        return reached

    def move_to(name: str, target: np.ndarray, closed: bool, tolerance: float = 0.025) -> bool:
        nonlocal observation
        reached = False
        for step in range(args.max_move_steps * 3):
            current = eef_position()
            error_base = controller.world_to_origin_frame(target) - controller.world_to_origin_frame(current)
            distance = float(np.linalg.norm(error_base))
            if distance <= tolerance:
                reached = True
                break
            command_limit = 0.20 if closed else 1.0
            command = np.clip(error_base / 0.05, -command_limit, command_limit)
            observation, _, _, _, _ = env.step(action(command, closed))
            record()
        phase_log.append(
            {
                "phase": name,
                "reached": reached,
                "steps": step + 1,
                "target_world": target.tolist(),
                "eef_world": eef_position().tolist(),
            }
        )
        print(f"{name}: {'reached' if reached else 'timeout'}")
        return reached

    def hold(name: str, closed: bool, steps: int) -> None:
        nonlocal observation
        for _ in range(steps):
            observation, _, _, _, _ = env.step(action(np.zeros(3), closed))
            record()
        phase_log.append({"phase": name, "steps": steps})

    def check_grasp(name: str) -> bool:
        grasped_now = bool(
            raw._check_grasp(robot.gripper["right"], raw.objects["obj"])
        )
        phase_log.append(
            {
                "phase": name,
                "grasped": grasped_now,
                "object_world": object_position().tolist(),
                "eef_world": eef_position().tolist(),
                "object_eef_distance": float(
                    np.linalg.norm(object_position() - eef_position())
                ),
            }
        )
        print(f"{name}: {'grasped' if grasped_now else 'lost'}")
        return grasped_now

    def move_base_to(name: str, target_xy: np.ndarray, closed: bool) -> bool:
        nonlocal observation
        reached = False
        for step in range(args.max_move_steps * 3):
            base_world = np.asarray(raw._get_observations(force_update=True)["robot0_base_pos"])
            world_error = np.r_[target_xy - base_world[:2], 0.0]
            local_error = controller.origin_ori.T @ world_error
            distance = float(np.linalg.norm(world_error[:2]))
            if distance <= 0.04:
                reached = True
                break
            base_command = np.zeros(4)
            base_command[:2] = np.clip(local_error[:2] / 0.20, -0.7, 0.7)
            base_command[3] = 0.5
            observation, _, _, _, _ = env.step(
                action(np.zeros(3), closed, base_command, base_mode=True)
            )
            record()
        phase_log.append(
            {
                "phase": name,
                "reached": reached,
                "steps": step + 1,
                "target_base_xy_world": target_xy.tolist(),
                "base_world": base_world.tolist(),
            }
        )
        print(f"{name}: {'reached' if reached else 'timeout'}")
        return reached

    try:
        record()
        initial_object = object_position()
        destination_fixture = raw.cab if args.destination == "cabinet" else raw.sink
        interior = next(iter(destination_fixture.get_int_sites(relative=False).values()))
        p0, px, py, pz = (np.asarray(point) for point in interior)
        width_vector = px - p0
        depth_vector = py - p0
        height_vector = pz - p0
        depth_direction = depth_vector / np.linalg.norm(depth_vector)
        if args.destination == "cabinet":
            destination_center = (
                p0
                + 0.5 * width_vector
                + args.cabinet_depth_fraction * depth_vector
                + 0.35 * height_vector
            )
            destination_front = (
                p0
                + 0.5 * width_vector
                - 0.12 * depth_direction
                + 0.35 * height_vector
            )
            destination_retreat = destination_front - 0.15 * depth_direction
        else:
            destination_center = (
                p0 + 0.5 * width_vector + 0.5 * depth_vector + 0.65 * height_vector
            )
            destination_front = destination_center + np.array([0.0, 0.0, 0.20])
            destination_retreat = destination_front + np.array([0.0, 0.0, 0.15])

        if args.grasp_mode == "side":
            approach_direction = initial_object - eef_position()
            approach_direction[2] = 0.0
            approach_direction /= np.linalg.norm(approach_direction)
            local_x_world = np.array([0.0, 0.0, 1.0])
            local_z_world = approach_direction
            local_y_world = np.cross(local_z_world, local_x_world)
            side_orientation = np.column_stack(
                [local_x_world, local_y_world, local_z_world]
            )
            grasp_center = initial_object + np.array(
                [0.0, 0.0, args.grasp_height_offset]
            )
            move_to_pose(
                "side_approach_object",
                grasp_center - approach_direction * 0.20,
                side_orientation,
                False,
            )
            move_to_pose(
                "side_contact_object",
                object_position()
                + np.array([0.0, 0.0, args.grasp_height_offset])
                - approach_direction * args.side_clearance,
                side_orientation,
                False,
                position_tolerance=0.012,
            )
        else:
            move_to("approach_object", initial_object + [0.0, 0.0, 0.18], False)
            move_to(
                "descend_to_object",
                object_position() + [0.0, 0.0, args.grasp_height_offset],
                False,
                tolerance=0.010,
            )
        hold("close_gripper", True, 35)

        grasped = bool(raw._check_grasp(robot.gripper["right"], raw.objects["obj"]))
        phase_log.append({"phase": "verify_grasp", "grasped": grasped})
        print(f"verify_grasp: {'grasped' if grasped else 'not grasped'}")

        lift_target = eef_position() + np.array([0.0, 0.0, 0.22])
        move_to("lift_object", lift_target, True)
        check_grasp("verify_grasp_after_lift")
        object_lifted = bool(object_position()[2] > initial_object[2] + 0.08)
        phase_log.append({"phase": "verify_lift", "object_lifted": object_lifted})
        print(f"verify_lift: {'lifted' if object_lifted else 'not lifted'}")
        move_to(f"approach_{args.destination}", destination_front, True)
        check_grasp(f"verify_grasp_at_{args.destination}_front")
        move_to(f"move_inside_{args.destination}", destination_center, True)
        check_grasp(f"verify_grasp_inside_{args.destination}")
        check_grasp("verify_grasp_before_release")
        hold("open_gripper", False, 35)
        move_to(f"retreat_from_{args.destination}", destination_retreat, False)
        hold("settle", False, 20)

        success = bool(raw._check_success())
        report = {
            "task": f"PickPlaceCounterTo{args.destination.title()}",
            "seed": args.seed,
            "object_group": args.object_group,
            "uses_ground_truth_pose": True,
            "grasp_mode": args.grasp_mode,
            "object_scale": args.object_scale,
            "cabinet_depth_fraction": args.cabinet_depth_fraction,
            "destination": args.destination,
            "success": success,
            "initial_object_world": initial_object.tolist(),
            "final_object_world": object_position().tolist(),
            "destination_target_world": destination_center.tolist(),
            "destination_retreat_world": destination_retreat.tolist(),
            "phases": phase_log,
            "video": str(video_path.resolve()),
        }
        report_path = args.output / "gt_pick_place_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"RoboCasa GT grasp loop: {'PASS' if success else 'INCOMPLETE'}")
        print(f"Report: {report_path.resolve()}")
        print(f"Video: {video_path.resolve()}")
    finally:
        writer.release()
        env.close()

    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()





