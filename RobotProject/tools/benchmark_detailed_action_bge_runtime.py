"""Record cold-load and warm-inference timing for the BGE detailed-action entry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.detailed_action_runtime import DetailedActionParser


def benchmark(instruction: str, device: str = "cpu") -> dict:
    started = time.perf_counter_ns()
    parser = DetailedActionParser(semantic_backend="bge", device=device)
    loaded = time.perf_counter_ns()
    first = parser.resolve(instruction)
    first_done = time.perf_counter_ns()
    second = parser.resolve(instruction)
    second_done = time.perf_counter_ns()
    return {
        "instruction": instruction,
        "device": device,
        "result_type": second.get("result_type"),
        "label": second.get("label"),
        "cold_load_seconds": (loaded - started) / 1e9,
        "first_inference_seconds": (first_done - loaded) / 1e9,
        "warm_inference_seconds": (second_done - first_done) / 1e9,
        "first_result": first,
        "second_result": second,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", default="请把牙刷立放进漱口杯")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/detailed_action_bge/runtime_timing.json"))
    args = parser.parse_args()
    result = benchmark(args.instruction, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"first_result", "second_result"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
