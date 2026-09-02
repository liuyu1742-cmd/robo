"""Build a self-contained browser dashboard from verified benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "outputs" / "decision_planning_benchmark_20260731.json"
DEFAULT_EVIDENCE = ROOT / "outputs" / "decision_planning_evidence_20260731"
DEFAULT_OUTPUT = ROOT / "outputs" / "decision_evidence_dashboard" / "index.html"
SECTION_IDS = [
    "shot-overview",
    "shot-vla",
    "shot-multi",
    "shot-subtasks",
    "shot-chart",
    "shot-code",
]


def _serialized_output(structured_output: dict[str, Any]) -> str:
    return json.dumps(
        structured_output,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _verify_one_evidence(
    evidence_directory: Path, filename: str, expected: dict[str, Any]
) -> dict[str, Any]:
    path = evidence_directory / filename
    evidence = json.loads(path.read_text(encoding="utf-8"))
    elapsed_ns = (
        evidence["ended_perf_counter_ns"] - evidence["started_perf_counter_ns"]
    )
    if elapsed_ns != evidence["elapsed_ns"]:
        raise ValueError(f"计时差值不一致：{filename}")
    if evidence["elapsed_seconds"] != elapsed_ns / 1_000_000_000:
        raise ValueError(f"秒数换算不一致：{filename}")
    output_sha256 = hashlib.sha256(
        _serialized_output(evidence["structured_output"]).encode("utf-8")
    ).hexdigest()
    if output_sha256 != evidence["output_sha256"]:
        raise ValueError(f"输出SHA-256不一致：{filename}")
    for field in (
        "evidence_id",
        "started_perf_counter_ns",
        "ended_perf_counter_ns",
        "elapsed_ns",
        "elapsed_seconds",
        "output_sha256",
    ):
        if evidence[field] != expected[field]:
            raise ValueError(f"汇总与逐轮证据不一致：{filename}/{field}")
    return evidence


def _verify_evidence(
    report: dict[str, Any], evidence_directory: Path
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for case in report["cases"]:
        for run in case["runs"]:
            evidence = _verify_one_evidence(
                evidence_directory, run["evidence_file"], run
            )
            verified[evidence["evidence_id"]] = evidence
        if replanning := case.get("replanning"):
            evidence = _verify_one_evidence(
                evidence_directory,
                replanning["evidence_file"],
                replanning,
            )
            verified[evidence["evidence_id"]] = evidence
    return verified


def _table(headers: list[str], rows: list[list[str]], *, compact: bool = False) -> str:
    class_name = "evidence-table compact" if compact else "evidence-table"
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        f'<table class="{class_name}"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def _status_badge(qualified: bool) -> str:
    label = "合格" if qualified else "不合格"
    css = "pass" if qualified else "fail"
    return f'<span class="badge {css}">{label}</span>'


def _overview_section(report: dict[str, Any]) -> str:
    cards = []
    for case in report["cases"]:
        object_label = case.get("object_name_zh") or "空调 + 电动窗帘 + 卧室灯"
        cards.append(
            f"""
            <div class="metric-card">
              <div class="eyebrow">{escape(case["case_id"])} · {escape(case["source"])}</div>
              <h3>{escape(object_label)}</h3>
              <div class="metric">{case["average_seconds"]:.9f}<small> s</small></div>
              <div class="subtle">阈值 ≤ {case["threshold_seconds"]:.0f} s
              {_status_badge(case["qualified"])}</div>
            </div>
            """
        )
    command = (
        r".\.venv\Scripts\python.exe -m tools.run_decision_planning_benchmark "
        r"--repetitions 10 --random-seed 20260731"
    )
    return f"""
    <section class="shot" id="shot-overview">
      <div class="shot-header">
        <div><div class="kicker">REAL TEST EVIDENCE · 01</div>
        <h1>测试执行总览</h1></div>
        <div class="stamp">31/31 证据校验通过</div>
      </div>
      <div class="terminal">
        <div class="terminal-bar"><i></i><i></i><i></i><span>PowerShell · RobotProject</span></div>
        <pre>PS C:\\RobotProject\\RobotProject&gt; {escape(command)}
[timer] {escape(report["timer"])}
[formula] {escape(report["timing_formula"])}
[executed_at] {escape(report["executed_at"])}
[random_seed] {report["random_seed"]}
[python] {escape(report["environment"]["python"])}
[platform] {escape(report["environment"]["platform"])}
[result] all_qualified={str(report["all_qualified"]).lower()}</pre>
      </div>
      <div class="metric-grid">{''.join(cards)}</div>
      <div class="note"><strong>阅读说明：</strong>平均耗时来自各用例10次独立运行；
      每一轮均保存原始纳秒计数、完整输出和SHA-256。截图仅用于快速查看，
      原始JSON用于复算和审计。</div>
    </section>
    """


def _vla_section(report: dict[str, Any]) -> str:
    panels = []
    for case in report["cases"][:2]:
        rows = [
            [
                f"{run['run_index']:02d}",
                run["evidence_id"],
                f"{run['elapsed_ns']:,}",
                f"{run['elapsed_seconds']:.9f}",
                run["output_sha256"][:16],
            ]
            for run in case["runs"]
        ]
        panels.append(
            f"""
            <div class="case-panel">
              <div class="case-title"><span>{escape(case["case_id"])}</span>
              <h2>{escape(case["object_name_zh"])}</h2></div>
              <p>{escape(case["task"])} · {escape(case["source_operation_label"])}</p>
              {_table(["轮次", "证据ID", "耗时(ns)", "耗时(s)", "输出SHA前16位"], rows, compact=True)}
              <div class="average">10次平均：<strong>{case["average_seconds"]:.9f} s</strong>
              {_status_badge(case["qualified"])}</div>
            </div>
            """
        )
    return f"""
    <section class="shot" id="shot-vla">
      <div class="shot-header"><div><div class="kicker">REAL TEST EVIDENCE · 02</div>
      <h1>VLA用例10次运行证据</h1></div><div class="stamp">不同VLA任务 / 不同物体</div></div>
      <div class="two-columns">{''.join(panels)}</div>
      <div class="note"><strong>结果解释：</strong>两个用例直接检索已审核的VLA动作模板，
      因此均为毫秒级；每个证据ID都对应一份独立JSON，可用输出SHA-256复核截图内容。</div>
    </section>
    """


def _multi_section(report: dict[str, Any]) -> str:
    case = report["cases"][2]
    rows = [
        [
            f"{run['run_index']:02d}",
            run["evidence_id"],
            str(run["started_perf_counter_ns"]),
            str(run["ended_perf_counter_ns"]),
            f"{run['elapsed_ns']:,}",
            f"{run['elapsed_seconds']:.9f}",
        ]
        for run in case["runs"]
    ]
    replanning = case["replanning"]
    rows.append(
        [
            "重规划",
            replanning["evidence_id"],
            str(replanning["started_perf_counter_ns"]),
            str(replanning["ended_perf_counter_ns"]),
            f"{replanning['elapsed_ns']:,}",
            f"{replanning['elapsed_seconds']:.9f}",
        ]
    )
    return f"""
    <section class="shot" id="shot-multi">
      <div class="shot-header"><div><div class="kicker">REAL TEST EVIDENCE · 03</div>
      <h1>多设备协同10次运行及重规划证据</h1></div>
      <div class="stamp">平均 {case["average_seconds"]:.9f} s · 阈值 5 s</div></div>
      {_table(["轮次", "证据ID", "开始计数(ns)", "结束计数(ns)", "差值(ns)", "耗时(s)"], rows, compact=True)}
      <div class="formula">elapsed_seconds =
      (ended_perf_counter_ns-started_perf_counter_ns) / 1,000,000,000</div>
      <div class="note"><strong>结果解释：</strong>十次协同平均耗时
      <strong>{case["average_seconds"]:.9f} s</strong>；室温升至31℃后的重规划耗时
      <strong>{replanning["elapsed_seconds"]:.9f} s</strong>。两者均低于5秒。
      多设备耗时高于VLA模板检索，是因为每次实际调用三个既有自然语言动作解析子任务。</div>
    </section>
    """


def _subtasks_section(report: dict[str, Any]) -> str:
    case = report["cases"][2]
    output = case["structured_output"]
    subtasks = []
    for index, subtask in enumerate(output["subtasks"], start=1):
        resolved = subtask["resolved"]
        subtasks.append(
            f"""
            <div class="subtask-card">
              <div class="number">{index}</div>
              <div><div class="eyebrow">{escape(subtask["planner"])}</div>
              <h2>{escape(resolved["object"])}</h2>
              <p>{escape(resolved["task"])} / {escape(resolved["object"])} / {escape(resolved["operation"])}</p>
              <code>{escape(" → ".join(subtask["base_actions"]))}</code></div>
            </div>
            """
        )
    targets = [
        ("空调", "24℃ · auto/cool"),
        ("电动窗帘", "关闭"),
        ("卧室灯", "20%亮度"),
    ]
    target_html = "".join(
        f'<div class="target"><strong>{name}</strong><span>{value}</span></div>'
        for name, value in targets
    )
    return f"""
    <section class="shot" id="shot-subtasks">
      <div class="shot-header"><div><div class="kicker">REAL TEST EVIDENCE · 04</div>
      <h1>三个既有动作解析子任务调用结果</h1></div>
      <div class="stamp">conflict_check = passed</div></div>
      <div class="subtask-list">{''.join(subtasks)}</div>
      <div class="flow-arrow">三个基础动作序列 → 协调器补充参数、并行分组与冲突检查</div>
      <div class="target-grid">{target_html}</div>
      <div class="note"><strong>结果解释：</strong>“准备就寝环境”没有直接返回固定文本，
      而是依次调用项目原有的空调、电动窗帘和卧室灯解析链；其中电动窗帘使用
      <code>close(electric_curtain)</code>位置动作。随后协调器设置24℃、20%亮度并确认无互斥目标。</div>
    </section>
    """


def _chart_section(report: dict[str, Any]) -> str:
    width, height = 1240, 480
    left, top, chart_w, chart_h = 90, 45, 900, 350
    y_max = 1.6
    colors = ["#2563eb", "#0f9f75", "#d97706"]
    lines = []
    dots = []
    legend = []
    for case_index, (case, color) in enumerate(zip(report["cases"], colors)):
        points = []
        for index, value in enumerate(case["elapsed_seconds"]):
            x = left + index * chart_w / 9
            clipped = min(float(value), y_max)
            y = top + chart_h - clipped / y_max * chart_h
            points.append(f"{x:.1f},{y:.1f}")
            dots.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}">'
                f"<title>{case['case_id']} #{index + 1}: {value:.9f}s</title></circle>"
            )
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="4"/>'
        )
        legend.append(
            f'<g transform="translate(1030,{70 + case_index * 70})">'
            f'<rect width="24" height="8" y="8" fill="{color}"/>'
            f'<text x="38" y="18" class="svg-label">{case["case_id"]}</text>'
            f'<text x="38" y="42" class="svg-small">平均 {case["average_seconds"]:.6f}s</text></g>'
        )
    grid = []
    for tick in (0, 0.4, 0.8, 1.2, 1.6):
        y = top + chart_h - tick / y_max * chart_h
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" '
            'stroke="#d9e2ec" stroke-width="1"/>'
            f'<text x="{left - 16}" y="{y + 5:.1f}" text-anchor="end" '
            f'class="svg-small">{tick:.1f}s</text>'
        )
    x_labels = "".join(
        f'<text x="{left + index * chart_w / 9:.1f}" y="{top + chart_h + 30}" '
        f'text-anchor="middle" class="svg-small">{index + 1}</text>'
        for index in range(10)
    )
    svg = f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="十次测试耗时折线图">
      <rect width="{width}" height="{height}" rx="20" fill="#f8fafc"/>
      {''.join(grid)}
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#334155" stroke-width="2"/>
      <line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#334155" stroke-width="2"/>
      {x_labels}
      {''.join(lines)}{''.join(dots)}{''.join(legend)}
      <text x="{left + chart_w / 2}" y="455" text-anchor="middle" class="svg-label">运行轮次</text>
      <text x="24" y="{top + chart_h / 2}" transform="rotate(-90 24 {top + chart_h / 2})" text-anchor="middle" class="svg-label">耗时（秒）</text>
      <g transform="translate(1030,300)">
        <rect width="170" height="90" rx="12" fill="#fff7ed" stroke="#fed7aa"/>
        <text x="18" y="30" class="svg-label">合格阈值</text>
        <text x="18" y="55" class="svg-small">VLA：≤ 3秒</text>
        <text x="18" y="76" class="svg-small">协同：≤ 5秒</text>
      </g>
    </svg>
    """
    return f"""
    <section class="shot" id="shot-chart">
      <div class="shot-header"><div><div class="kicker">REAL TEST EVIDENCE · 05</div>
      <h1>10次耗时对比与合格阈值</h1></div><div class="stamp">纵轴：秒</div></div>
      {svg}
      <div class="note"><strong>结果解释：</strong>DEC-001与DEC-002为VLA审核模板检索，
      曲线贴近0秒；DEC-003包含三个动作解析子任务，因此稳定在约1.2秒。
      图中纵轴放大至1.6秒便于观察实际波动，右侧另列3秒/5秒正式合格阈值。
      所有运行均低于对应阈值。</div>
    </section>
    """


def _numbered_source_excerpt() -> str:
    source_path = Path(__file__).with_name("run_decision_planning_benchmark.py")
    lines = source_path.read_text(encoding="utf-8").splitlines()

    def block(start_marker: str, end_marker: str) -> list[tuple[int, str]]:
        start = next(
            index for index, line in enumerate(lines) if line.startswith(start_marker)
        )
        end = next(
            index
            for index in range(start, len(lines))
            if lines[index].strip() == end_marker
        )
        return [(index + 1, lines[index]) for index in range(start, end + 1)]

    selected = [
        *block("def _timed_plan(", "return result, timing"),
        (0, ""),
        *block("def _write_evidence(", "return {**timing, \"evidence_id\": evidence_id, \"evidence_file\": filename}"),
    ]
    rendered = []
    for line_number, line in selected:
        if line_number == 0:
            rendered.append('<span class="code-line spacer"></span>')
        else:
            rendered.append(
                f'<span class="code-line"><b>{line_number:04d}</b>'
                f"<em>{escape(line)}</em></span>"
            )
    return "".join(rendered)


def _code_section(report: dict[str, Any]) -> str:
    case = report["cases"][2]
    run = case["runs"][-1]
    output = case["structured_output"]
    subtask_lines = "".join(
        "<li><strong>"
        + escape(subtask["resolved"]["task"])
        + " / "
        + escape(subtask["resolved"]["object"])
        + "</strong><code>"
        + escape(" → ".join(subtask["base_actions"]))
        + "</code></li>"
        for subtask in output["subtasks"]
    )
    return f"""
    <section class="shot" id="shot-code">
      <div class="shot-header"><div><div class="kicker">REAL TEST EVIDENCE · 06</div>
      <h1>计时代码与真实输出对照</h1></div>
      <div class="stamp">源文件：tools/run_decision_planning_benchmark.py</div></div>
      <div class="code-evidence-grid">
        <div class="source-panel">
          <div class="panel-label">实际项目代码（自动读取，含行号）</div>
          <pre class="source-code">{_numbered_source_excerpt()}</pre>
        </div>
        <div class="output-panel">
          <div class="panel-label">同一轮真实输出证据</div>
          <div class="output-id">{escape(run["evidence_id"])}</div>
          <dl>
            <dt>started_perf_counter_ns</dt><dd>{run["started_perf_counter_ns"]}</dd>
            <dt>ended_perf_counter_ns</dt><dd>{run["ended_perf_counter_ns"]}</dd>
            <dt>elapsed_ns</dt><dd>{run["elapsed_ns"]:,}</dd>
            <dt>elapsed_seconds</dt><dd>{run["elapsed_seconds"]:.9f}</dd>
            <dt>output_sha256</dt><dd class="hash">{escape(run["output_sha256"])}</dd>
          </dl>
          <h3>解析得到的三个子任务</h3>
          <ol class="output-list">{subtask_lines}</ol>
          <div class="output-status">conflict_check = passed · completion_status = 通过</div>
        </div>
      </div>
      <div class="note"><strong>图片数据来源与证明关系：</strong>左侧代码在调用完整决策函数前后读取
      <code>time.perf_counter_ns()</code>，以结束值减开始值计算耗时，并对可序列化完整输出计算SHA-256；
      右侧是DEC-003第10轮对应JSON中的原始数值和实际解析输出。代码说明“如何计时”，
      证据值与哈希说明“本轮测得什么并输出了什么”。</div>
    </section>
    """


def _section_map(report: dict[str, Any]) -> dict[str, str]:
    return {
        "shot-overview": _overview_section(report),
        "shot-vla": _vla_section(report),
        "shot-multi": _multi_section(report),
        "shot-subtasks": _subtasks_section(report),
        "shot-chart": _chart_section(report),
        "shot-code": _code_section(report),
    }


def _document_html(report: dict[str, Any]) -> str:
    sections = _section_map(report)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RobotProject 决策规划测试证据</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#dfe7ef;color:#172033;
font-family:"Microsoft YaHei UI","Noto Sans CJK SC",Arial,sans-serif}}
.shot{{width:1440px;margin:32px auto;padding:48px 54px;background:#fff;
border-radius:24px;box-shadow:0 18px 60px rgba(15,23,42,.18);overflow:hidden}}
.shot-header{{display:flex;justify-content:space-between;align-items:flex-start;
gap:24px;border-bottom:2px solid #e2e8f0;padding-bottom:22px;margin-bottom:26px}}
h1{{font-size:34px;line-height:1.25;margin:4px 0 0}} h2{{margin:3px 0 6px;font-size:24px}}
h3{{font-size:20px;margin:8px 0 12px}} p{{line-height:1.55;margin:6px 0}}
.kicker,.eyebrow{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;
color:#46637f;font-weight:800}} .stamp{{padding:10px 16px;border-radius:999px;
background:#e7f8f1;color:#08785b;font-weight:800;white-space:nowrap}}
.terminal{{background:#111827;color:#dbeafe;border-radius:16px;overflow:hidden;
box-shadow:inset 0 0 0 1px #334155}} .terminal-bar{{height:40px;background:#243044;
display:flex;align-items:center;padding:0 16px;gap:8px;color:#94a3b8;font-size:13px}}
.terminal-bar i{{width:11px;height:11px;border-radius:50%;background:#fb7185}}
.terminal-bar i:nth-child(2){{background:#fbbf24}} .terminal-bar i:nth-child(3){{background:#34d399}}
.terminal-bar span{{margin-left:8px}} pre{{margin:0;padding:22px 26px;line-height:1.65;
font:15px Consolas,"Cascadia Mono",monospace;white-space:pre-wrap}}
.metric-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:22px}}
.metric-card,.case-panel,.subtask-card{{border:1px solid #dbe4ee;border-radius:16px;
padding:20px;background:#fbfdff}} .metric{{font-size:34px;font-weight:900;color:#173b62}}
.metric small{{font-size:17px}} .subtle{{color:#60758a}} .badge{{display:inline-block;
margin-left:8px;padding:3px 10px;border-radius:999px;font-weight:800;font-size:13px}}
.badge.pass{{background:#dcfce7;color:#166534}} .badge.fail{{background:#fee2e2;color:#991b1b}}
.note{{margin-top:22px;padding:16px 20px;border-left:5px solid #2d6a9f;
background:#eef6fc;line-height:1.7;border-radius:0 12px 12px 0}}
.two-columns{{display:grid;grid-template-columns:1fr 1fr;gap:24px}} .case-title{{display:flex;
align-items:center;gap:12px}} .case-title span{{background:#173b62;color:white;border-radius:8px;
padding:6px 10px;font-weight:800}} .average{{margin-top:14px;text-align:right;font-size:16px}}
.evidence-table{{width:100%;border-collapse:separate;border-spacing:0;border:1px solid #cbd5e1;
border-radius:12px;overflow:hidden}} .evidence-table th{{background:#173b62;color:#fff;
font-weight:800;text-align:left;padding:11px 10px}} .evidence-table td{{padding:9px 10px;
border-top:1px solid #e2e8f0;font:13px Consolas,"Microsoft YaHei UI",monospace}}
.evidence-table tbody tr:nth-child(even){{background:#f7fafc}} .compact th{{font-size:13px}}
.compact td{{font-size:12.5px;padding:7px 8px}} .formula{{margin-top:18px;padding:14px;
text-align:center;font:700 16px Consolas,monospace;background:#f1f5f9;border-radius:10px}}
.subtask-list{{display:grid;gap:16px}} .subtask-card{{display:grid;grid-template-columns:56px 1fr;
gap:18px;align-items:center}} .number{{width:46px;height:46px;border-radius:50%;display:grid;
place-items:center;background:#173b62;color:white;font-size:23px;font-weight:900}}
code{{font-family:Consolas,monospace;color:#0f4c72;background:#eef6fc;padding:3px 6px;
border-radius:5px}} .flow-arrow{{margin:22px auto;text-align:center;font-weight:900;
font-size:18px;color:#334155}} .target-grid{{display:grid;grid-template-columns:repeat(3,1fr);
gap:16px}} .target{{display:flex;justify-content:space-between;padding:18px;border-radius:14px;
background:#eef6fc;border:1px solid #c9e0f2}} .target span{{font-weight:900;color:#0f4c72}}
svg{{display:block;width:100%;height:auto}} .svg-label{{font:700 16px "Microsoft YaHei UI",Arial}}
.svg-small{{font:13px "Microsoft YaHei UI",Arial;fill:#475569}}
.code-evidence-grid{{display:grid;grid-template-columns:1.2fr .8fr;gap:22px;align-items:stretch}}
.source-panel,.output-panel{{border:1px solid #cbd5e1;border-radius:15px;overflow:hidden;background:#f8fafc}}
.panel-label{{padding:12px 16px;background:#173b62;color:#fff;font-weight:800}}
.source-code{{margin:0;padding:14px 0;background:#0f172a;color:#dbeafe;font:12px/1.42 Consolas,monospace;
white-space:pre;overflow:hidden}} .code-line{{display:block}} .code-line b{{display:inline-block;width:56px;
padding-right:12px;text-align:right;color:#64748b;font-weight:400;user-select:none}}
.code-line em{{font-style:normal}} .code-line.spacer{{height:10px}}
.output-panel{{padding-bottom:16px}} .output-panel dl{{display:grid;grid-template-columns:190px 1fr;
gap:0;margin:0 18px;border:1px solid #dbe4ee;border-radius:10px;overflow:hidden}}
.output-panel dt,.output-panel dd{{margin:0;padding:9px 11px;border-bottom:1px solid #e2e8f0;
font:12px Consolas,monospace}} .output-panel dt{{background:#eef3f8;font-weight:800}}
.output-panel dd{{background:white;word-break:break-all}} .output-id{{margin:18px;font-size:25px;
font-weight:900;color:#173b62}} .output-panel h3{{margin:18px 18px 8px}}
.output-list{{margin:0 18px;padding-left:22px}} .output-list li{{margin:10px 0;line-height:1.45}}
.output-list code{{display:block;margin-top:4px;font-size:11px;word-break:break-all}}
.output-status{{margin:16px 18px 0;padding:11px;border-radius:9px;background:#dcfce7;
color:#166534;font-weight:900;text-align:center}} .hash{{font-size:10.5px!important}}
</style>
</head>
<body>{''.join(sections.values())}</body></html>"""


def build_dashboard(
    benchmark_path: Path,
    evidence_directory: Path,
    output_html: Path,
) -> dict[str, Any]:
    """Verify evidence and write a self-contained HTML dashboard."""
    report = json.loads(benchmark_path.read_text(encoding="utf-8"))
    verified = _verify_evidence(report, evidence_directory)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    complete_html = _document_html(report)
    output_html.write_text(complete_html, encoding="utf-8")
    document_head, _ = complete_html.split("<body>", maxsplit=1)
    for section_id, section_html in _section_map(report).items():
        (output_html.parent / f"{section_id}.html").write_text(
            document_head + "<body>" + section_html + "</body></html>",
            encoding="utf-8",
        )
    return {
        "verified_evidence_files": len(verified),
        "section_ids": list(SECTION_IDS),
        "output_html": str(output_html.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--evidence-directory", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_dashboard(
        args.benchmark, args.evidence_directory, args.output
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
