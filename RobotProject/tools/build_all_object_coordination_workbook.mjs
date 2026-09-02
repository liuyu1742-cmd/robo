import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(.:)/, "$1")), "..");
const outputDir = path.join(root, "outputs", "019fb5d7-4c5a-7e23-bb89-f7ec997f7068");
const catalogPath = path.join(outputDir, "all_object_coordination_catalog.json");
const outputPath = path.join(outputDir, "全物体多设备协调联动命令与动作解析审核表.xlsx");
const previewDir = path.join(outputDir, "workbook_previews");
const catalog = JSON.parse(await fs.readFile(catalogPath, "utf8"));

const workbook = Workbook.create();
const commandSheet = workbook.worksheets.add("多设备联动命令表");
const detailSheet = workbook.worksheets.add("逐物体动作解析测试表");
const auditSheet = workbook.worksheets.add("覆盖审核表");

const COLORS = {
  navy: "#17365D", blue: "#1F4E78", teal: "#0F6B78", green: "#548235",
  lightBlue: "#DDEBF7", lightTeal: "#DDEBF7", lightGreen: "#E2F0D9",
  lightOrange: "#FCE4D6", lightYellow: "#FFF2CC", red: "#C00000",
  lightRed: "#F4CCCC", gray: "#E7E6E6", white: "#FFFFFF", text: "#1F2937",
};

function styleTitle(sheet, endColumn, title, subtitle) {
  sheet.mergeCells(`A1:${endColumn}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${endColumn}1`).format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 16 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${endColumn}1`).format.rowHeight = 34;
  sheet.mergeCells(`A2:${endColumn}2`);
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${endColumn}2`).format = {
    fill: COLORS.lightBlue,
    font: { color: COLORS.text, size: 10 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange(`A2:${endColumn}2`).format.rowHeight = 32;
  sheet.showGridLines = false;
}

function styleHeader(sheet, range) {
  sheet.getRange(range).format = {
    fill: COLORS.blue,
    font: { bold: true, color: COLORS.white, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#B4C6E7" },
  };
  sheet.getRange(range).format.rowHeight = 36;
}

function numbered(values) {
  return values.map((value, index) => `${index + 1}. ${value}`).join("\n");
}

// Sheet 1: coordination commands.
const commandHeaders = [
  "联动编号", "场景类别", "场景名称", "多设备协调命令", "涉及任务数", "涉及物体数",
  "原有物体", "VLA物体", "执行阶段摘要", "执行方式", "环境或状态触发条件", "安全与冲突规则",
  "预期总体输出", "合格判定标准", "审核状态", "审核意见",
];
styleTitle(
  commandSheet,
  "P",
  "全物体多设备协调联动命令表",
  `共 ${catalog.summary.scenario_count} 条真实生活场景命令；覆盖原有对象 ${catalog.summary.original_unique_objects}/123、VLA对象 ${catalog.summary.vla_objects}/82、总对象 ${catalog.summary.total_objects}/205。审核状态初始均为“待审核”。`,
);
commandSheet.getRange("A4:P4").values = [commandHeaders];
const commandRows = catalog.scenarios.map((row) => [
  row.scenario_id, row.category, row.name, row.command, row.task_count, row.object_count,
  row.original_objects.join("；"), row.vla_objects.join("；"), row.execution_phases,
  row.execution_modes, row.trigger_condition, row.conflict_rules, row.expected_output,
  row.qualification, row.review_status, row.review_comment,
]);
commandSheet.getRangeByIndexes(4, 0, commandRows.length, commandHeaders.length).values = commandRows;
const commandEnd = 4 + commandRows.length;
commandSheet.tables.add(`A4:P${commandEnd}`, true, "CoordinationCommandsTable").style = "TableStyleMedium2";
styleHeader(commandSheet, "A4:P4");
commandSheet.freezePanes.freezeRows(4);
commandSheet.freezePanes.freezeColumns(3);
commandSheet.getRange(`A5:P${commandEnd}`).format = { verticalAlignment: "top", wrapText: true, font: { size: 9, color: COLORS.text } };
commandSheet.getRange(`E5:F${commandEnd}`).format.numberFormat = "0";
commandSheet.getRange(`O5:O${commandEnd}`).dataValidation = { rule: { type: "list", values: ["待审核", "通过", "需修改"] } };
commandSheet.getRange(`O5:O${commandEnd}`).conditionalFormats.add("containsText", { text: "待审核", format: { fill: COLORS.lightYellow } });
commandSheet.getRange(`O5:O${commandEnd}`).conditionalFormats.add("containsText", { text: "通过", format: { fill: COLORS.lightGreen } });
commandSheet.getRange(`O5:O${commandEnd}`).conditionalFormats.add("containsText", { text: "需修改", format: { fill: COLORS.lightRed, font: { color: COLORS.red } } });
const commandWidths = [13, 14, 18, 48, 10, 10, 30, 30, 36, 24, 30, 38, 38, 38, 12, 24];
commandWidths.forEach((width, index) => commandSheet.getRangeByIndexes(0, index, commandEnd, 1).format.columnWidth = width);
commandSheet.getRange(`A5:P${commandEnd}`).format.rowHeight = 76;

// Sheet 2: per-object action parsing tests.
const detailHeaders = [
  "联动编号", "场景类别", "测试场景工况", "测试操作步骤", "预期输出结果", "物体来源", "任务类别",
  "物体ID", "中文名称", "子任务指令", "识别操作", "原动作序列", "新增细分动作序列", "前置条件",
  "细分控制参数", "执行阶段", "执行模式", "依赖物体", "目标状态", "冲突规则", "合格判定", "审核状态", "审核意见",
];
styleTitle(
  detailSheet,
  "W",
  "逐物体动作解析测试表",
  "每个对象单独一行；原动作序列保持项目现状，新增细分动作序列为额外输出。VLA细分动作直接采用已审核目录。",
);
detailSheet.getRange("A4:W4").values = [detailHeaders];
const detailRows = catalog.object_details.map((row) => [
  row.scenario_id, row.category, row.test_condition, row.test_steps, row.expected_output,
  row.source, row.task_name_zh, row.object_id, row.object_name_zh, row.subtask_instruction,
  row.operation, numbered(row.original_actions), numbered(row.detailed_actions), row.prerequisites,
  row.control_parameters, row.execution_phase, row.execution_mode, row.dependencies, row.target_state,
  row.conflict_rule, row.qualification, row.review_status, row.review_comment,
]);
detailSheet.getRangeByIndexes(4, 0, detailRows.length, detailHeaders.length).values = detailRows;
const detailEnd = 4 + detailRows.length;
detailSheet.tables.add(`A4:W${detailEnd}`, true, "ObjectActionParsingTable").style = "TableStyleMedium2";
styleHeader(detailSheet, "A4:W4");
detailSheet.freezePanes.freezeRows(4);
detailSheet.freezePanes.freezeColumns(2);
detailSheet.getRange(`A5:W${detailEnd}`).format = { verticalAlignment: "top", wrapText: true, font: { size: 9, color: COLORS.text } };
detailSheet.getRange(`V5:V${detailEnd}`).dataValidation = { rule: { type: "list", values: ["待审核", "通过", "需修改"] } };
detailSheet.getRange(`V5:V${detailEnd}`).conditionalFormats.add("containsText", { text: "待审核", format: { fill: COLORS.lightYellow } });
detailSheet.getRange(`V5:V${detailEnd}`).conditionalFormats.add("containsText", { text: "通过", format: { fill: COLORS.lightGreen } });
detailSheet.getRange(`V5:V${detailEnd}`).conditionalFormats.add("containsText", { text: "需修改", format: { fill: COLORS.lightRed, font: { color: COLORS.red } } });
for (let rowIndex = 0; rowIndex < catalog.object_details.length; rowIndex += 1) {
  const excelRow = rowIndex + 5;
  const fill = catalog.object_details[rowIndex].source === "VLA" ? COLORS.lightOrange : COLORS.lightBlue;
  detailSheet.getRange(`F${excelRow}:I${excelRow}`).format.fill = fill;
}
const detailWidths = [12, 14, 30, 46, 60, 12, 18, 16, 16, 38, 26, 42, 52, 34, 34, 24, 12, 32, 30, 34, 38, 12, 24];
detailWidths.forEach((width, index) => detailSheet.getRangeByIndexes(0, index, detailEnd, 1).format.columnWidth = width);
detailSheet.getRange(`A5:W${detailEnd}`).format.rowHeight = 118;

// Sheet 3: formula-driven coverage audit.
const auditHeaders = [
  "物体来源", "任务类别", "物体ID", "中文名称", "对应联动编号", "出现次数", "子任务指令", "原动作序列",
  "细分动作序列", "目标状态", "冲突规则", "覆盖状态", "审核状态", "审核意见",
];
styleTitle(
  auditSheet,
  "N",
  "全对象覆盖审核表",
  "覆盖状态和必填项检查由公式引用“逐物体动作解析测试表”计算；用户修改明细后可重新打开Excel自动复核。",
);
auditSheet.getRange("A3:B3").values = [["覆盖指标", "公式结果"]];
auditSheet.getRange("A4:A6").values = [["原有对象覆盖"], ["VLA对象覆盖"], ["总对象覆盖"]];
auditSheet.getRange("B4:B6").formulas = [[`=COUNTIFS($A$9:$A$${8 + catalog.coverage.length},"原有目录",$L$9:$L$${8 + catalog.coverage.length},"完整")&"/123"`], [`=COUNTIFS($A$9:$A$${8 + catalog.coverage.length},"VLA",$L$9:$L$${8 + catalog.coverage.length},"完整")&"/82"`], [`=COUNTIF($L$9:$L$${8 + catalog.coverage.length},"完整")&"/205"`]];
auditSheet.getRange("A3:B6").format = { borders: { preset: "all", style: "thin", color: "#A6A6A6" }, wrapText: true };
auditSheet.getRange("A3:B3").format = { fill: COLORS.green, font: { bold: true, color: COLORS.white }, horizontalAlignment: "center" };
auditSheet.getRange("A4:A6").format = { fill: COLORS.lightGreen, font: { bold: true } };
auditSheet.getRange("B4:B6").format = { fill: "#F3F8EE", font: { bold: true, color: COLORS.green }, horizontalAlignment: "center" };
auditSheet.getRange("A8:N8").values = [auditHeaders];
const auditRows = catalog.coverage.map((row) => [
  row.source, row.task, row.object_id, row.object_name_zh, row.scenario_ids.join("；"),
  null, null, null, null, null, null, null, row.review_status, row.review_comment,
]);
auditSheet.getRangeByIndexes(8, 0, auditRows.length, auditHeaders.length).values = auditRows;
const auditEnd = 8 + auditRows.length;
for (let index = 0; index < catalog.coverage.length; index += 1) {
  const row = index + 9;
  const detailSource = `'逐物体动作解析测试表'!$F$5:$F$${detailEnd}`;
  const detailId = `'逐物体动作解析测试表'!$H$5:$H$${detailEnd}`;
  auditSheet.getRange(`F${row}`).formulas = [[`=COUNTIFS(${detailSource},A${row},${detailId},C${row})`]];
  auditSheet.getRange(`G${row}`).formulas = [[`=IF(COUNTIFS(${detailSource},A${row},${detailId},C${row},'逐物体动作解析测试表'!$J$5:$J$${detailEnd},"<>")>0,"有","缺失")`]];
  auditSheet.getRange(`H${row}`).formulas = [[`=IF(COUNTIFS(${detailSource},A${row},${detailId},C${row},'逐物体动作解析测试表'!$L$5:$L$${detailEnd},"<>")>0,"有","缺失")`]];
  auditSheet.getRange(`I${row}`).formulas = [[`=IF(COUNTIFS(${detailSource},A${row},${detailId},C${row},'逐物体动作解析测试表'!$M$5:$M$${detailEnd},"<>")>0,"有","缺失")`]];
  auditSheet.getRange(`J${row}`).formulas = [[`=IF(COUNTIFS(${detailSource},A${row},${detailId},C${row},'逐物体动作解析测试表'!$S$5:$S$${detailEnd},"<>")>0,"有","缺失")`]];
  auditSheet.getRange(`K${row}`).formulas = [[`=IF(COUNTIFS(${detailSource},A${row},${detailId},C${row},'逐物体动作解析测试表'!$T$5:$T$${detailEnd},"<>")>0,"有","缺失")`]];
  auditSheet.getRange(`L${row}`).formulas = [[`=IF(AND(F${row}>0,G${row}="有",H${row}="有",I${row}="有",J${row}="有",K${row}="有"),"完整","不完整")`]];
}
auditSheet.tables.add(`A8:N${auditEnd}`, true, "CoverageAuditTable").style = "TableStyleMedium4";
styleHeader(auditSheet, "A8:N8");
auditSheet.freezePanes.freezeRows(8);
auditSheet.freezePanes.freezeColumns(4);
auditSheet.getRange(`A9:N${auditEnd}`).format = { verticalAlignment: "top", wrapText: true, font: { size: 9, color: COLORS.text } };
auditSheet.getRange(`F9:F${auditEnd}`).format.numberFormat = "0";
auditSheet.getRange(`M9:M${auditEnd}`).dataValidation = { rule: { type: "list", values: ["待审核", "通过", "需修改"] } };
auditSheet.getRange(`L9:L${auditEnd}`).conditionalFormats.add("containsText", { text: "完整", format: { fill: COLORS.lightGreen, font: { color: COLORS.green } } });
auditSheet.getRange(`L9:L${auditEnd}`).conditionalFormats.add("containsText", { text: "不完整", format: { fill: COLORS.lightRed, font: { color: COLORS.red, bold: true } } });
auditSheet.getRange(`G9:K${auditEnd}`).conditionalFormats.add("containsText", { text: "缺失", format: { fill: COLORS.lightRed, font: { color: COLORS.red } } });
const auditWidths = [12, 20, 18, 18, 18, 10, 12, 12, 14, 12, 12, 12, 12, 24];
auditWidths.forEach((width, index) => auditSheet.getRangeByIndexes(0, index, auditEnd, 1).format.columnWidth = width);
auditSheet.getRange(`A9:N${auditEnd}`).format.rowHeight = 32;

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

for (const config of [
  { sheetName: "多设备联动命令表", range: "A1:H12", file: "01_commands_left.png" },
  { sheetName: "多设备联动命令表", range: "I1:P12", file: "02_commands_right.png" },
  { sheetName: "逐物体动作解析测试表", range: "A1:H10", file: "03_object_details_test.png" },
  { sheetName: "逐物体动作解析测试表", range: "I1:P10", file: "04_object_details_actions.png" },
  { sheetName: "逐物体动作解析测试表", range: "Q1:W10", file: "05_object_details_controls.png" },
  { sheetName: "覆盖审核表", range: "A1:N18", file: "06_coverage_audit.png" },
]) {
  const preview = await workbook.render({ sheetName: config.sheetName, range: config.range, scale: 1.2, format: "png" });
  await fs.writeFile(path.join(previewDir, config.file), new Uint8Array(await preview.arrayBuffer()));
}

const inspections = {};
for (const [name, range] of [["多设备联动命令表", "A1:P8"], ["逐物体动作解析测试表", "A1:W7"], ["覆盖审核表", "A1:N12"]]) {
  inspections[name] = (await workbook.inspect({ kind: "region", sheetId: name, range, maxChars: 4500 })).ndjson;
}
const errorScan = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
await fs.writeFile(path.join(outputDir, "workbook_inspection.json"), JSON.stringify({ inspections, errorScan: errorScan.ndjson }, null, 2), "utf8");
console.log(JSON.stringify({ outputPath, scenarioRows: commandRows.length, detailRows: detailRows.length, auditRows: auditRows.length, previews: 6, errorScan: errorScan.ndjson }, null, 2));
