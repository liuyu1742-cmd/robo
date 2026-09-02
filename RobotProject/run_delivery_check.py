"""Run the project delivery checks in one command.

This script is meant for PyCharm Terminal use. It runs the same local checks
that support the current delivery claim:

1. unit tests
2. acceptance check
3. 15+120 dataset/code coverage audit
4. P1 evidence readiness check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_JSON_REPORT = ROOT / "datasets" / "delivery_check_report.json"
DEFAULT_MD_REPORT = ROOT / "docs" / "DELIVERY_CHECK_REPORT.md"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def python_command(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def build_delivery_plan() -> dict:
    return {
        "format": "delivery_check_plan_v1",
        "json_report": relative(DEFAULT_JSON_REPORT),
        "markdown_report": relative(DEFAULT_MD_REPORT),
        "steps": [
            {
                "name": "unit_tests",
                "title": "单元测试",
                "command": python_command("-m", "unittest", "discover", "-s", "tests"),
            },
            {
                "name": "acceptance_check",
                "title": "总验收检查",
                "command": python_command("run_acceptance_check.py"),
                "output": "docs/ACCEPTANCE_CHECK_REPORT.md",
            },
            {
                "name": "dataset_code_coverage_15x120",
                "title": "15+120数据/代码证据审计",
                "command": python_command("tools/audit_dataset_code_coverage_15x120.py"),
                "output": "docs/DATASET_CODE_COVERAGE_15X120_REPORT.md",
            },
            {
                "name": "p1_evidence_readiness",
                "title": "P1本地数据证据就绪度",
                "command": python_command("tools/check_p1_evidence_readiness.py"),
                "output": "docs/P1_EVIDENCE_READINESS_REPORT.md",
            },
        ],
    }


def run_step(step: dict, *, dry_run: bool = False) -> dict:
    if dry_run:
        return {
            **step,
            "returncode": 0,
            "status": "skipped_dry_run",
            "stdout_tail": "",
            "stderr_tail": "",
        }

    completed = subprocess.run(
        step["command"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        **step,
        "returncode": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# 一键交付检查报告",
        "",
        f"- 运行时间：{report['generated_at']}",
        f"- 总体状态：{'通过' if report['passed'] else '未通过'}",
        f"- 检查项：{report['passed_steps']}/{report['total_steps']} 通过",
        "",
        "## 检查项",
        "",
        "| 检查 | 状态 | 说明/输出 |",
        "|---|---|---|",
    ]
    for step in report["steps"]:
        output = step.get("output", "")
        output_text = f"`{output}`" if output else ""
        lines.append(f"| {step['title']} | `{step['status']}` | {output_text} |")

    failed = [step for step in report["steps"] if step["status"] == "failed"]
    if failed:
        lines.extend(["", "## 失败项最后输出", ""])
        for step in failed:
            lines.extend(
                [
                    f"### {step['title']}",
                    "",
                    "```text",
                    (step.get("stderr_tail") or step.get("stdout_tail") or "").strip(),
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "## 结论",
                "",
                "本次本地交付检查全部通过。注意：P1报告如果显示图片和标注仍为0，只能说明模板就绪，不能说明P1真实数据集已完成。",
                "",
            ]
        )
    return "\n".join(lines)


def run_delivery_check(*, dry_run: bool = False) -> dict:
    plan = build_delivery_plan()
    steps = [run_step(step, dry_run=dry_run) for step in plan["steps"]]
    passed_steps = sum(1 for step in steps if step["returncode"] == 0)
    report = {
        "format": "delivery_check_report_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "passed": passed_steps == len(steps),
        "total_steps": len(steps),
        "passed_steps": passed_steps,
        "steps": steps,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned checks.")
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_delivery_check(dry_run=args.dry_run)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.md_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Use UTF-8 with BOM for the Markdown report so Windows PowerShell/PyCharm
    # terminals display Chinese text correctly when users open it with
    # Get-Content or simple editor defaults.
    args.md_report.write_text(markdown_report(report), encoding="utf-8-sig")
    print(json.dumps({k: report[k] for k in ["passed", "passed_steps", "total_steps"]}, ensure_ascii=False, indent=2))
    print(f"json: {args.json_report.resolve()}")
    print(f"markdown: {args.md_report.resolve()}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
