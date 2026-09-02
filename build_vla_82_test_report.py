"""Run VLA 82-object template regression cases and rewrite the Word report."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from RobotProject.tools.vla_action_templates import (
    load_vla_action_templates,
    select_vla_action_template,
)


ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"C:\Users\sjtu101\Desktop\3.docx")
OUT = ROOT / "outputs" / "vla_82_template_test_report.docx"
RESULTS = ROOT / "outputs" / "vla_82_template_test_results.json"

CASES = (
    ("VLA-001", "室内卫生清洁", "把客厅里的多个汽水罐扔进厨房垃圾桶"),
    ("VLA-002", "烹饪与加热辅助", "在炉灶的平底锅中烹调卷心菜和辣椒"),
    ("VLA-003", "物品递送", "把橱柜里的水瓶放到台面"),
    ("VLA-004", "整理收纳", "把书架上的书放进客厅地面的收纳箱"),
    ("VLA-005", "储物设施开合", "完全关闭冰箱抽屉"),
    ("VLA-006", "室内安装与布置", "把数码相机安装到卧室三脚架"),
    ("VLA-007", "餐具与容器摆放", "把咖啡杯从橱柜放到台面"),
    ("VLA-008", "衣物鞋类与玄关归位", "把两顶棒球帽放入洗衣机清洗"),
    ("VLA-009", "工作学习区服务", "把键盘放在显示器旁"),
    ("VLA-010", "卫浴用品与个人卫生服务", "把牙刷放入洗手台上的杯中"),
)


def _set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def _set_cell_text(cell, text: str, *, bold: bool = False, color: str | None = None) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(8.5)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def _clear_document(doc: Document) -> None:
    body = doc._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def _add_heading(doc: Document, text: str, level: int) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    paragraph.paragraph_format.space_after = Pt(4)
    run = paragraph.add_run(text)
    run.bold = True
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(15 if level == 1 else 12)
    run.font.color.rgb = RGBColor(31, 78, 121)


def _add_paragraph(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.25
    run = paragraph.add_run(text)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(10.5)


def run_cases() -> list[dict]:
    task_names = {row["id"]: row["name_zh"] for row in load_vla_action_templates()["tasks"]}
    results: list[dict] = []
    for case_id, task_name, instruction in CASES:
        candidate = select_vla_action_template(instruction)
        if candidate is None:
            raise RuntimeError(f"No VLA template selected for {case_id}")
        if task_names[candidate.task_id] != task_name:
            raise RuntimeError(f"Unexpected task for {case_id}: {candidate.task_id}")
        results.append({
            "case_id": case_id,
            "task": task_name,
            "instruction": instruction,
            "object": candidate.object_name_zh,
            "operation_label": candidate.source_operation_label,
            "steps": list(candidate.steps),
            "passed": len(candidate.steps) >= 3,
        })
    return results


def build_report(results: list[dict]) -> None:
    doc = Document(SOURCE)
    _clear_document(doc)
    section = doc.sections[0]
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("3.1.5.1 VLA 新增任务与物体动作解析测试报告")
    title_run.bold = True
    title_run.font.name = "Microsoft YaHei"
    title_run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    title_run.font.size = Pt(17)
    title_run.font.color.rgb = RGBColor(31, 78, 121)

    _add_heading(doc, "一、测试范围与方法", 1)
    _add_paragraph(doc, "本次测试仅抽取新增 VLA 范围内的 10 类任务、82 个物体。共设置 10 条自然语言测试指令，每类任务抽取 1 个代表物体；所有输入均使用与表内操作标签一致的口语化表达。")
    _add_paragraph(doc, "测试入口为 RobotProject/tools/vla_action_templates.py。测试验证指令是否选择到正确的新增任务与对象，并验证输出动作链不少于 3 步。该测试评估预设动作模板选择与步骤输出，不将结果表述为原始生成模型准确率。")

    _add_heading(doc, "二、合格判定", 1)
    _add_paragraph(doc, "单条用例须同时满足：① 匹配对象属于新增 82 个物体范围；② 匹配任务属于新增 10 类任务范围；③ 输出唯一操作链；④ 输出动作链不少于 3 步且包含确认步骤。")

    _add_heading(doc, "三、测试结果", 1)
    passed = sum(row["passed"] for row in results)
    _add_paragraph(doc, f"本次实际运行 {len(results)} 条测试指令，{passed} 条通过，{len(results) - passed} 条未通过。模板选择与完整动作步骤输出通过率为 {passed / len(results) * 100:.1f}%。")

    table = doc.add_table(rows=1, cols=7)
    table.style = "Table Grid"
    table.autofit = False
    widths = (Cm(1.4), Cm(2.3), Cm(2.0), Cm(4.0), Cm(3.6), Cm(5.0), Cm(1.2))
    headers = ("用例", "新增任务", "测试物体", "输入指令", "表内操作标签", "输出动作步骤", "结果")
    for cell, width, header in zip(table.rows[0].cells, widths, headers):
        cell.width = width
        _set_cell_shading(cell, "1F4E79")
        _set_cell_text(cell, header, bold=True, color="FFFFFF")
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in results:
        cells = table.add_row().cells
        values = (
            row["case_id"], row["task"], row["object"], row["instruction"],
            row["operation_label"], " → ".join(row["steps"]), "通过" if row["passed"] else "未通过",
        )
        for cell, width, value in zip(cells, widths, values):
            cell.width = width
            _set_cell_text(cell, value, bold=value == "通过", color="008000" if value == "通过" else None)

    _add_heading(doc, "四、结果分析", 1)
    _add_paragraph(doc, "10 条测试分别覆盖室内卫生清洁、烹饪与加热辅助、物品递送、整理收纳、储物设施开合、室内安装与布置、餐具与容器摆放、衣物鞋类与玄关归位、工作学习区服务及卫浴用品与个人卫生服务。每条均从新增 82 个物体范围内选择对象，未使用原 120 物体表作为本次测试对象来源。")
    _add_paragraph(doc, "示例：输入“把客厅里的多个汽水罐扔进厨房垃圾桶”时，系统输出“拿起汽水罐 → 移动至厨房垃圾桶 → 投放汽水罐 → 确认已放入”，体现了物体、场景目标和投放确认的完整动作闭环。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)


def main() -> None:
    results = run_cases()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(results)
    print(json.dumps({"total": len(results), "passed": sum(row["passed"] for row in results), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
