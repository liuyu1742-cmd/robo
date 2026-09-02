# RobotProject 当前项目状态（面向非专业读者）

> 文件日期：2026-07-04；依据任务图片最终口径更新：2026-07-07  
> 历史验收目标（2026-07-04 版本）：不少于15类家政任务、120种典型物体/设备、标准指令映射到标准动作序列，以及冻结测试集的整序列准确率不低于90%。  
> **当前权威范围（2026-07-17）**：15类任务、**123种去重物体**、**138条任务—物体操作关系**。下文的“120/120”均指历史标准指令测试中的**120条测试样例**，不是当前物体类别数量。
> 范围说明：本轮是软件验收，必须保留“视觉动作数据处理/特征提取/技能学习迁移”链路；准确率≥90%的指定测法是“标准指令→标准动作序列”。不要求机械臂控制、仿真成功率或真机验证，也不把“从原始视频同时猜动作、任务和物体”作为该90%指标的计算口径。
> 交付拆分：图片中的内容现在拆成两个任务分别交付。任务A是“15类任务、120物体/设备、动作解析准确率”；任务B是“网上整理对应公开数据集和代码”。两者互相关联，但验收材料分开保存。

## 一、先说结论

项目已完成视觉动作数据处理/特征提取链路，并建立15类任务、123种去重物体、138条任务—物体操作语义及“标准指令→标准动作序列”冻结测试。历史正式测试时只输入 instruction，120条历史测试样例整序列全部正确。

- 按“数据和算法原型”计算：约完成 **90%**。
- 按本轮图片任务的软件验收计算：约完成 **95%**。
- 15类任务框架、123种去重物体和138条任务—物体操作映射已经建立。
- 已建立480条冻结基准：训练240、验证120、测试120；每个集合都覆盖15类和120项。
- 历史标准测试整序列准确率 **100%（120/120 条测试样例）**，高于90%要求；它不表示“120种物体”，也不能替代当前自然语言盲测结果。
- 此前约2%是更困难的“从视频同时识别动作、任务和物体”指标，属于扩展研究，不影响本轮指定测法的通过结论。

通俗地说：教材、标准答案、冻结考卷和正式判分都已经完成；历史标准题库考试通过。当前目录已扩展为123种去重物体、138条操作关系；自然语言预评测尚未达到90%，不能外推为任意口语或任意视频同样100%。

## 二、完成等级是什么意思

为了避免“清单里有”被误解为“已经通过测试”，本轮软件范围把进度分成四级：

1. **已定义**：系统知道任务或物体叫什么，以及理论上要做哪些步骤。
2. **有数据/模型**：有训练样本，模型能够给出预测。
3. **冻结测试通过**：没有见过测试答案，整条动作序列仍能正确生成。
4. **扩展执行验证**：仿真或真实机器人可以执行；这是加分项，不是本轮必选项。

当前成果已经达到第3级：冻结测试通过。已有RoboCasa成果保留为扩展证明，不影响本轮纯软件验收。

## 三、逐项对照图片要求

| 图片要求 | 已完成 | 未完成 | 当前判断 |
|---|---|---|---|
| 不少于15类典型家政任务 | 15类均进入训练、验证和测试；测试每类8条 | 可继续增加更自然的人工口语表达 | **完成** |
| 120种物体/设备操作技能学习 | 建立120项覆盖清单，且120项均进入训练、验证和测试 | 不要求机械臂逐项实操；可扩展更多同义名称 | **完成** |
| 多动作解析准确率≥90% | 冻结测试120条整序列准确率100%，步骤准确率100% | 结果只适用于当前标准指令基准 | **通过** |
| 标准指令映射到标准动作序列 | 测试只输入instruction，模型独立输出完整动作序列；模板跨集合重叠为0；已新增人工同义指令盲测模板和覆盖120项的弹窗式手动输入测试入口 | 人工填写同义指令后还需正式跑盲测报告 | **通过，可继续加固** |
| 整理公开数据集和代码 | 已整理/使用EPIC-KITCHENS、HICO-DET、HOI4D标注、COCO/LVIS、YCB参考及Ultralytics、RoboCasa/MuJoCo | HOI4D完整RGB-D/手部轨迹尚未全部落地；许可证、版本和复现实验清单待完善 | **大部分完成** |

补充审计：已新增 `docs/DATASET_CODE_COVERAGE_15X120_REPORT.md`。按正式15+120清单核对，120项中107项能找到明确数据集或代码证据，13项暂缺明确证据；`datasets/object_operation_coverage_120.json` 已按正式 `meta/compliance_catalog_15x120.json` 同步，名称漂移已清零。P1三项 `video_doorbell`、`child_clothing`、`glucose_meter` 已建立本地数据证据模板，并补充20槽位最小样本计划、人工标注规范、质量检查清单和就绪度报告；当前就绪度结论是模板已就绪，但还未放入真实图片和标注。

补充拆分：已新增任务A和任务B的独立交付说明：`docs/TASK_A_ACTION_PARSING_DELIVERY.md` 与 `docs/TASK_B_PUBLIC_DATASET_CODE_DELIVERY.md`。任务B按“不自采图片，只从公开网站/公开数据集查找”的口径推进，并已新增13类公开来源映射和风险登记。

## 四、15类家政任务的详细进度

图片中的任务存在交叉，例如“地面清洁”又细分木地板和瓷砖。项目内部目前采用12个核心任务标签；下面为了对应验收，将材质和衣物流程拆成15个可检查场景。**本轮只要求这些场景具有标准指令和标准动作序列，不要求15套机器人控制策略。**

| 序号 | 目标场景 | 已经做到什么 | 还缺什么 | 当前等级 |
|---:|---|---|---|---|
| 1 | 普通地面清洁 | 有任务定义、动作词和公开数据映射 | 缺完整扫/拖轨迹及重复仿真测试 | 已定义/有数据 |
| 2 | 木地板清洁 | 已作为材质场景写入覆盖清单 | 缺材质识别、清洁剂选择和力度控制 | 已定义 |
| 3 | 瓷砖清洁 | 已列为独立验收场景 | 缺污渍识别、擦洗和避障验证 | 已定义 |
| 4 | 洗碗机/餐具清洁 | EPIC中有厨房动作；玻璃杯装载到水槽的RoboCasa物理闭环已通过 | 缺开门、完整清洗、启动和取出的连续成功率 | 有数据/单场景仿真通过 |
| 5 | 智能烹饪 | 有拿、放、切、开关等样本和序列模型 | 缺火候、食材状态、安全和完整菜品闭环 | 有数据/部分模型 |
| 6 | 食物和饮品递送 | 能表达“定位—抓取—移动—放置” | 缺液体防洒和动态避障验证 | 有序列/少量仿真 |
| 7 | 衣物整理 | 儿童服装、鞋、棉衣、羽绒制品已进入清单 | 缺柔性衣物姿态识别、折叠和分类放置 | 已定义 |
| 8 | 洗衣护理 | 洗衣机等设备有操作模板 | 未接真实设备，未验证材质和程序选择 | 已定义 |
| 9 | 晾晒与烘干 | 电动晾衣架已进入清单 | 缺展开、夹持、晾晒和设备控制闭环 | 已定义 |
| 10 | 家庭整理收纳 | 有整理任务数据；调料瓶整理入柜的RoboCasa物理闭环已通过 | 缺杂乱环境多物体规划和长期成功率 | 单场景仿真通过 |
| 11 | 取物与递送 | 缩放到夹爪可容纳尺寸的玻璃杯已完成抓取、抬升、运输和入柜 | 缺纯视觉定位、新物体泛化和真机验证 | 单场景仿真通过 |
| 12 | 垃圾收集处理 | 有任务标签和标准步骤 | 缺识别、装袋、分类、投放全流程 | 已定义/有数据 |
| 13 | 卫浴清洁 | 有类别和标准步骤框架 | 缺湿滑环境、安全接触和清洁效果检测 | 已定义 |
| 14 | 家电综合管理 | 灯、电视、空调、冰箱、洗衣机、净化器、电饭煲等有技能名称 | 多数是软件命令模板，未连接真实设备协议 | 已定义 |
| 15 | 智能安防巡检 | 门锁、摄像头、温度计、体重秤等进入清单 | 缺异常事件数据、隐私策略和真实设备联调 | 已定义 |

表中“仿真、控制、真机”缺口仅记录已有扩展工作的真实状态，不作为本轮图片任务的扣分项。按本轮验收，15类标准指令样例和冻结测试覆盖已经完成；后续缺口是更自然口语的人工盲测。

总体上：

- 任务名称和边界：基本完成。
- 自然语言指令和标准动作步骤：大部分完成。
- 视频样本：厨房、拿取、放置较多；清洁、衣物、安防、家电控制不足。
- 仿真执行：主要是抓取、移动、放置等基础技能。
- 真实机器人：没有覆盖15类任务的可验收结果。

## 五、120种物体和设备的详细进度

项目已建立 `datasets/object_operation_coverage_120.json`，记录“要覆盖什么、属于哪个任务、需要哪些技能”。这相当于课程目录已经列齐，不代表120项实操考试都通过。

### 已完成

- 120项物体/设备均有记录，覆盖地面材质、衣物、厨房用品、家具和智能家电。
- 建立名称归一化和操作映射。例如把“水杯/杯子”归到同一概念，再关联定位、抓取、移动、放置。
- 图片中点名的儿童服装、鞋、棉衣、羽绒制品、吸顶灯、阅读灯、电视、空调、冰箱、洗衣机、空气净化器、电动晾衣架、电饭煲、扫地机器人、门锁、摄像头、温度计和体重秤等，已纳入覆盖框架。

### 未完成

以下视觉检测和设备执行缺口属于后续扩展，不阻塞本轮标准指令解析验收；120项已经全部进入冻结指令训练、验证和测试覆盖。

- 当前家居YOLO检测器重点覆盖 **16类**，不能理解为摄像头已稳定识别120类。
- 本地14个EPIC视频主要是厨房场景，只包含120项中的一部分。
- 柔性衣物、透明/反光物体、小物体和外观相似物体仍然很难识别。
- 对电视、空调、门锁等设备，多数只有控制动作定义，没有真实品牌协议和安全验证。
- 尚无逐项记录“能否识别、定位、操作、失败恢复、仿真通过、真机通过”的120项验收表。

按本轮范围的准确表述是：**知识与操作映射120/120已建立，且120/120已纳入冻结指令测试覆盖。视觉识别和机器人实操属于扩展能力，不作为本轮≥90%指令解析验收的前置条件。**

## 六、公开数据集和代码

| 数据/代码 | 用途 | 当前状态 |
|---|---|---|
| EPIC-KITCHENS | 学习第一视角厨房视频中的拿、放、开、关等动作 | 已选14个视频，形成3,786个片段、5,674个动作段 |
| HICO-DET | 学习“人—物体—动作”关系 | 已整理12,350条统一样本 |
| HOI4D | 室内人手与物体交互参考 | 已整理2,973条标注；完整RGB-D和轨迹未全部落地 |
| COCO/LVIS | 通用物体检测和类别体系 | 已用于检测和类别参考，不能替代家政专用数据 |
| YCB | 常见抓取物体参考 | 已用于轻量物体/类别映射 |
| Ultralytics YOLO | 物体检测和视觉特征 | 已接入训练、视频处理流程 |
| RoboCasa/MuJoCo | 家庭机器人仿真 | 已有基础抓取、放置和行为克隆原型 |

统一动作数据约 **20,997条**：EPIC 5,674条、HICO-DET 12,350条、HOI4D 2,973条；训练/验证/测试约15,524/2,313/3,160条。

公开数据只是训练材料，不能单独证明项目达标。最终还要登记版本、许可证、下载来源、使用范围和复现命令。

## 七、视觉动作处理与技能学习链路

1. 从115个候选EPIC-KITCHENS视频中选择体量可控的视频。
2. Tier A的14个视频已完成下载和校验。
3. 长视频已切成3,786个短片段，并形成5,674个动作段。
4. 每个动作段抽取3帧，使用家居YOLO骨干生成512维视觉特征。
5. 5,674/5,674个动作段特征提取成功，解码失败为0。
6. 视觉特征已接入动作Transformer，训练“下一步做什么”。
7. `final_demo.py`可读取预计算特征，也可从新视频在线提取并尝试预测条件。
8. 已修复 `electric_drying_rack` 等物体词表缺项，并做词表差集检查。

因此，“视频→特征→动作模型”的程序通路已经打通，可作为图片第一句“基于视觉的人类动作捕获、解析和技能学习迁移”的工程证据。它是任务组成部分，但不使用视频联合识别率计算图片指定的90%主指标；90%仍按标准指令映射标准动作序列计算。

## 八、本轮准确率应该怎样计算

### 正式主指标

测试时只给系统一条标准指令或未见过的同义表达，例如“将水杯放到桌上”。系统必须独立输出完整序列：

`定位水杯 → 抓取水杯 → 移动到桌旁/桌上方 → 放到桌面 → 检查是否放稳`

人工预设的标准动作序列是测试正确答案，不需要系统从视频生成该标准答案。正式主指标为：

`整序列准确率 = 完全正确的测试指令数 ÷ 测试指令总数`

“完全正确”要求步骤集合正确、顺序正确、物体参数正确、目标位置正确。最终要求该指标不低于90%，并辅助报告步骤准确率、遗漏率、顺序错误率、物体准确率和目标位置准确率。

### 2026-07-06正式冻结测试结果

- 基准总量：480条（训练240、验证120、测试120）。
- 任务覆盖：15/15；物体/设备覆盖：120/120；三个集合均为全覆盖。
- 指令模板跨训练/验证/测试重叠：0。
- 测试输入：只提供instruction；task、object、target和标准actions在生成完成后才用于判分。
- 指令字段解析准确率：100%。
- 整序列准确率：**100%（120/120）**。
- 步骤/动词/参数准确率：100%；遗漏率、额外步骤率、顺序错误率：0%。
- ≥90%要求：**通过**。

该结果证明当前系统能完成受控标准指令基准，不证明任意自由口语、全新任务或原始视频也有100%。

### 受控动作序列模型

- 最佳教师强制验证序列准确率：**99.22%**。
- “教师强制”相当于考试时前面正确步骤已经告诉模型，只让它猜下一步。
- 它证明模型学到了动作顺序，但不证明模型能独立看懂原始视频。

### 视觉消融

- 有视觉特征：EPIC验证序列准确率约98.379%。
- 去掉视觉特征：约98.093%。
- 视觉带来的提升只有约 **0.286个百分点**。

这说明视觉有帮助，但高分主要来自任务、物体和前序动作等结构化提示。

### 更接近真实使用的纯视觉结果

- 动作（动词）Top-1：**30.79%**
- 任务Top-1：**77.22%**
- 物体Top-1：**8.96%**
- 动作+任务+物体同时正确：**2.00%**

这些纯视觉结果不再决定本轮图片验收是否通过。它们只说明视频自主理解仍有研究空间。

### 示例：“将水杯放到桌上”

系统已经可以把标准答案写成：

`定位水杯 → 抓取水杯 → 移动到桌旁/桌上方 → 放到桌面 → 检查是否放稳`

已经完成的是标准动作序列的数据表达、预测框架和冻结指令测试；当前正式测试整序列准确率为100%，高于≥90%要求。

## 九、已经完成的主要成果

1. 任务、物体、动作和目标位置的统一数据格式。
2. 120项物体/设备操作覆盖清单。
3. 15个任务演示场景及其到现有核心任务标签的映射。
4. EPIC、HICO-DET、HOI4D标注共约20,997条统一样本。
5. EPIC Tier A 14个视频的下载、校验、切片和源视频清理，数据量约4.39 GiB。
6. 5,674个动作段的512维视觉特征。
7. 接入视觉特征的动作Transformer及有/无视觉对照实验。
8. 从视频特征预测动作、任务和物体的视觉条件分类器。
9. 可读取视频并输出预测动作序列的演示入口。
10. 三个代表任务已通过RoboCasa官方物理成功判定：水杯取放入柜、调料瓶整理入柜、玻璃杯装载到水槽；均使用真值物体位姿。
11. 三个RoboCasa固定场景已改用“目标实例IoU”复测，不再把同类别误检算作通过；在置信度≥0.30且目标IoU≥0.30时，三项任务均至少有一个相机真正框中目标（3/3）。7个目标可见视角命中5个，视角召回71.4%，仍未达到90%。
12. 已打通MuJoCo实例分割自动标注并完成两轮定向微调：第一批30个场景/90张图，第二批12个独立困难场景/36张图，合计126张。第二轮合并验证集总体mAP50为0.957，cup为0.918、bottle为0.995；固定测试的同类别误检从5个降到1个。
13. 已审计原有480条标准指令基准：15类和120项在训练、验证、测试中均全覆盖，模板跨集合重叠为0。
14. 已恢复并修复专用指令动作Transformer训练/加载入口，验证集在第10轮达到整序列100%。
15. 已建立严格评估器，保证生成阶段只读取instruction，并保存120条逐条预测和错误记录。
16. 正式测试整序列准确率100%（120/120），通过图片要求的≥90%门槛；一键验收脚本已按确认口径返回通过。
17. 已新增独立人工同义指令盲测流程：`prepare_blind_paraphrase_test.py` 可导出120条人工填写表，填写后可转成盲测JSON并复用正式评测器。
18. 已新增覆盖120项的手动输入测试入口：`manual_instruction_test.py`。直接运行会弹出输入框；输入“清洗衣物”“把杯子洗了”“打开阅读灯”“检查门锁安全”“把手机拿过来”等自然语言，会先在15类/120项中匹配任务和物体，再输出对应动作序列并与标准序列比对。
19. 已新增15+120数据集/代码证据审计脚本：`tools/audit_dataset_code_coverage_15x120.py`。当前结论为107项有明确证据、13项暂缺明确证据。
20. 已新增并运行正式清单同步脚本：`tools/sync_object_operation_coverage_120.py`。`datasets/object_operation_coverage_120.json` 与正式120清单已对齐，名称漂移为0。
21. 已为P1三项建立本地数据证据模板：`datasets/p1_evidence/video_doorbell/`、`datasets/p1_evidence/child_clothing/`、`datasets/p1_evidence/glucose_meter/`。每项包含README、图片目录、标注目录、来源登记表、标注模板、20槽位最小样本计划、人工标注规范和质量检查清单。
22. 已新增P1就绪度检查：`tools/check_p1_evidence_readiness.py`。当前报告显示3/3模板就绪、真实图片0、标注文件0、可进入训练复核0，因此状态应表述为“模板就绪，待补真实样本”，不能表述为“P1数据集已完成”。
23. 已新增一键交付检查入口：`run_delivery_check.py`。它会连续运行单元测试、总验收检查、15+120数据/代码证据审计和P1就绪度检查，并生成 `docs/DELIVERY_CHECK_REPORT.md` 与 `datasets/delivery_check_report.json`；当前4/4检查通过。
24. 已按用户要求把两个任务分开交付：任务A为动作解析准确率，任务B为公开数据集/代码整理。任务B新增 `docs/PUBLIC_DATASET_SOURCE_MAP_13.md`、`datasets/public_dataset_source_map_13.json` 和 `docs/PUBLIC_DATASET_RISK_REGISTER.md`，并明确不使用自采图片。
25. 已按“不自采，只用公开网站/公开数据集”的要求下载13类第一批Wikimedia Commons公开图片证据：`datasets/public_dataset_13_images/`。当前13/13类均有成功manifest记录，本地图片文件共22张；每条记录保留页面、下载链接、许可证和风险备注。该批数据仍需人工筛选和标注后才能用于训练。
26. 已新增“公开数据集网站筛选下载计划”：`docs/PUBLIC_DATASET_SITE_DOWNLOAD_PLAN_13.md` 和 `datasets/public_dataset_site_download_plan_13.csv`。明确优先从COCO-Stuff、Fashionpedia、Open Images和Wikimedia Commons筛选下载；小红书、淘宝、Facebook、Instagram只作外观参考，不进入训练集。
27. 已新增13类公开图片筛选、标注和训练准备度入口：`datasets/public_dataset_13_images/public_image_review_sheet.csv`、`tools/build_public_annotation_templates.py`、`tools/check_public_training_readiness.py`。用户已完成第一批人工筛选：22张候选图中保留16张、剔除6张、待筛选0张；其中10类有保留图，`video_doorbell`、`window_glass`、`glucose_meter` 当前保留图仍为0。
28. 已生成13类补下载计划：`docs/PUBLIC_IMAGE_TOPUP_PLAN.md` 和 `datasets/public_image_topup_plan.csv`。以每类20张保留图为目标，当前总缺口244张，优先补3个保留图为0的类别。

## 十、尚未完成的主要工作

### P0：图片核心验收

1. 15类、120项冻结指令基准：已完成。
2. 只输入instruction并独立生成完整序列：已完成。
3. 整序列、步骤、遗漏、顺序、动词和参数指标：已完成。
4. 模板跨集合泄漏检查：已完成，重叠数为0。
5. 视觉动作数据处理、特征提取和技能学习链路：已有可运行代码与证据。
6. 整序列准确率≥90%：已通过，当前为100%。

### P1：不阻塞本轮验收的增强项

7. 视频自主识别的动作+任务+物体联合准确率仍低，后续可继续提升，但不作为本轮≥90%的计算口径。
8. 当前家居检测器只重点覆盖16类；扩展到120类视觉检测属于增强项。
9. RoboCasa三任务、3D定位、机械臂、ROS2、真机和智能家电协议均属于后续执行层，不阻塞本轮软件验收。

### P2：仍建议完成的数据与报告维护

10. HOI4D完整RGB、深度、手部姿态和相机参数尚未全部下载处理。
11. 清洁、衣物、安防和设备控制视觉样本少于厨房动作。
12. 需要继续清除旧报告中的乱码、过时路径和旧模型引用。
13. 公开数据集与代码的版本、许可证、来源、用途和一键复现命令仍可进一步补齐。
14. 15+120数据/代码证据仍有13项缺明确公开数据或代码证据：`wood_floor`、`tile_floor`、`marble_floor`、`laminate_floor`、`vinyl_floor`、`video_doorbell`、`child_clothing`、`window_glass`、`window_frame`、`screen_window`、`balcony_door`、`window_sill`、`glucose_meter`。
15. 旧覆盖表名称漂移已修复；后续只需继续为13项缺口补公开数据或代码证据。
16. 已为13项缺口生成补齐路线：`docs/MISSING_13_DATASET_EVIDENCE_PLAN.md`，其中P1优先项为 `video_doorbell`、`child_clothing`、`glucose_meter`。
17. P1三项已补“模板级证据”和“最小样本计划”：每项20个必需样本槽位，合计60个槽位；但还未补真实图片、标注和训练结果。对外只能说“已建立采集/标注模板和检查口径，待补样本”。
18. 已新增13类补齐排期：`docs/MISSING_13_COMPLETION_SCHEDULE.md`。模板级补齐预计1个工作日；小样本真实证据补齐预计3到5个工作日；训练级补齐预计1到2周，前提是允许使用自采或公开可用图片并完成来源登记。

## 十一、各模块进度估计

| 模块 | 估计进度 | 外行解释 |
|---|---:|---|
| 任务/物体/动作定义 | 90% | 要做什么基本说清楚了 |
| 公开数据整理 | 80% | 主要数据已整理，仍缺部分完整时序数据 |
| 120项知识与操作映射 | 100% | 目录齐全，且120项均已进入正式指令测试 |
| 15类指令和标准序列 | 100% | 15类均进入冻结基准并逐类通过 |
| 120项指令测试覆盖 | 100% | 120项在训练、验证和测试中均全覆盖 |
| EPIC视频处理与视觉特征 | 90% | Tier A流水线完整 |
| 物体视觉感知 | 50% | 仍仅16类重点检测；目标实例任务级3/3，但可见视角召回仅71.4%，尚未达到90% |
| 端到端视频动作理解 | 35% | 程序能跑，但物体和联合识别很弱 |
| 标准技能序列规划 | 100% | 测试时只输入指令，120/120整序列正确 |
| 视觉动作处理与特征 | 85% | 视频切片、R3D特征和视觉条件模型已可运行，需整理复现报告 |
| 指令解析正式基准 | 100% | 480条、15类、120项、模板无跨集合重叠 |
| 独立测试与最终验收 | 100% | 正式测试100%，一键验收按确认口径通过 |
| 仿真/机器人执行（范围外） | 不计入 | 已有三项仿真成果，但不是本轮必选项 |

百分比不能简单平均。按最终确认的图片任务口径，数据/模型原型约90%，本轮核心软件验收已通过；剩余约5%主要是公开数据/代码许可证、版本和复现说明的交付整理。

## 十二、现在能演示和不能宣称的内容

### 现在可以演示

- 从已有或相似的厨房视频中提取视觉特征。
- 在给定任务/物体等条件时预测下一动作或动作序列。
- 展示“定位—抓取—移动—放置”的标准步骤。
- 展示15类场景和120项物体/设备覆盖清单。
- 运行冻结标准指令测试并复现120/120整序列正确结果。
- 直接运行 `manual_instruction_test.py` 弹出输入框，输入自然语言后在120项操作中匹配并输出动作序列。已验证示例包括“清洗衣物”“把杯子洗了”“打开阅读灯”“检查门锁安全”“把手机拿过来”。
- 作为扩展演示，在RoboCasa/MuJoCo中展示三个物理闭环任务。

### 现在不能可靠宣称

- 任意自由口语、错别字或完全新任务也能保持100%。
- 99.22%的教师强制分数等于正式指令解析分数；正式分数来自独立冻结测试的100%。
- 任意家务视频都能以90%以上准确率被解析；这不是本轮90%指标的含义。
- 120项正式指令测试通过等于机器人已经能在真实世界操作120种物体。

## 十三、下一步任务

核心准确率任务已经完成。下一步转为交付加固：

1. 让人工填写 `datasets/instruction_blind_paraphrase_template.csv` 的 `human_instruction` 列，然后生成并运行盲测报告，检查当前100%是否能在独立同义口语中保持≥90%。
2. 整理EPIC-KITCHENS、HICO-DET、HOI4D、COCO/LVIS、YCB及相关代码的版本、许可证、来源链接和用途。
3. 若继续补P1数据证据，优先按 `datasets/p1_evidence/*/minimum_sample_plan.csv` 放入真实图片，并同步填写来源、标注和复核状态。
4. 继续补齐公开数据集与代码的版本、许可证、来源链接、用途和复现说明。
5. 若继续补P1真实数据，先按60个槽位补图片和标注，再重新运行 `run_delivery_check.py`。
6. 保留视频自主理解、YOLO和RoboCasa作为扩展成果，不再阻塞本轮交付。

## 十四、主要证据文件

- 120项覆盖清单：`datasets/object_operation_coverage_120.json`
- 15个演示任务：`datasets/final_demo_cases_15tasks.json`
- 统一动作数据：`datasets/project_action_dataset.jsonl`
- EPIC视觉特征：`datasets/epic_kitchens/processed/tier_a_visual_features.npz`
- EPIC片段汇总：`datasets/epic_kitchens/processed/project_tier_a_clip_index_summary.json`
- 动作模型：`models/action_transformer_project_visual.pt`
- 视觉条件分类器：`models/epic_visual_condition_classifier.pt`
- 视觉条件报告：`datasets/epic_visual_condition_report.json`
- 视觉消融报告：`datasets/project_model_visual_ablation_report.json`
- 冻结指令基准：`datasets/standard_instruction_action_benchmark.json`
- 指令动作模型：`models/instruction_action_transformer_v1.pt`
- 指令训练报告：`datasets/instruction_action_training_report.json`
- 正式测试JSON：`datasets/instruction_action_benchmark_report.json`
- 正式测试文档：`docs/INSTRUCTION_ACTION_BENCHMARK_REPORT.md`
- 15+120数据/代码证据审计JSON：`datasets/dataset_code_coverage_15x120_report.json`
- 15+120数据/代码证据审计文档：`docs/DATASET_CODE_COVERAGE_15X120_REPORT.md`
- 15+120数据/代码证据审计脚本：`tools/audit_dataset_code_coverage_15x120.py`
- 120项覆盖清单同步脚本：`tools/sync_object_operation_coverage_120.py`
- 13项缺口补齐路线JSON：`datasets/missing_13_dataset_evidence_plan.json`
- 13项缺口补齐路线文档：`docs/MISSING_13_DATASET_EVIDENCE_PLAN.md`
- 13项缺口补齐路线脚本：`tools/plan_missing_dataset_evidence_13.py`
- 13项缺口补齐排期：`docs/MISSING_13_COMPLETION_SCHEDULE.md`
- 任务A独立交付说明：`docs/TASK_A_ACTION_PARSING_DELIVERY.md`
- 任务B独立交付说明：`docs/TASK_B_PUBLIC_DATASET_CODE_DELIVERY.md`
- 13类公开数据源映射文档：`docs/PUBLIC_DATASET_SOURCE_MAP_13.md`
- 13类公开数据源映射JSON：`datasets/public_dataset_source_map_13.json`
- 13类公开数据集网站下载计划：`docs/PUBLIC_DATASET_SITE_DOWNLOAD_PLAN_13.md`
- 13类公开数据集网站下载计划CSV：`datasets/public_dataset_site_download_plan_13.csv`
- 公开数据集风险登记：`docs/PUBLIC_DATASET_RISK_REGISTER.md`
- 13类公开图片第一批目录：`datasets/public_dataset_13_images/`
- 13类公开图片manifest：`datasets/public_dataset_13_images/public_commons_13_manifest.csv`
- 13类公开图片下载报告：`docs/PUBLIC_COMMONS_13_DOWNLOAD_REPORT.md`
- 13类公开图片下载汇总JSON：`datasets/public_commons_13_download_summary.json`
- 13类公开图片下载脚本：`tools/download_public_commons_13.py`
- 13类公开图片汇总脚本：`tools/summarize_public_commons_13.py`
- 13类公开图片人工筛选表：`datasets/public_dataset_13_images/public_image_review_sheet.csv`
- 13类公开图片标注模板：`datasets/public_dataset_13_images/public_annotation_template.csv`
- 13类公开图片标注模板报告：`docs/PUBLIC_ANNOTATION_TEMPLATE_REPORT.md`
- 13类公开图片训练准备度报告：`docs/PUBLIC_TRAINING_READINESS_REPORT.md`
- 13类公开图片标注模板脚本：`tools/build_public_annotation_templates.py`
- 13类公开图片训练准备度脚本：`tools/check_public_training_readiness.py`
- 13类公开图片补下载计划：`docs/PUBLIC_IMAGE_TOPUP_PLAN.md`
- 13类公开图片补下载CSV：`datasets/public_image_topup_plan.csv`
- 13类公开图片补下载脚本：`tools/plan_public_image_topup.py`
- P1三项本地证据模板目录：`datasets/p1_evidence/`
- P1三项本地证据模板汇总：`docs/P1_EVIDENCE_TEMPLATE_SUMMARY.md`
- P1三项模板生成脚本：`tools/build_p1_evidence_templates.py`
- P1三项就绪度JSON：`datasets/p1_evidence_readiness_report.json`
- P1三项就绪度文档：`docs/P1_EVIDENCE_READINESS_REPORT.md`
- P1三项就绪度检查脚本：`tools/check_p1_evidence_readiness.py`
- 人工同义指令盲测填写表：`datasets/instruction_blind_paraphrase_template.csv`
- 人工同义指令盲测准备入口：`prepare_blind_paraphrase_test.py`
- 弹窗式120项手动指令测试入口：`manual_instruction_test.py`
- 手动指令测试报告：`datasets/manual_instruction_test_report.json`
- 一键验收报告：`docs/ACCEPTANCE_CHECK_REPORT.md`
- 一键交付检查入口：`run_delivery_check.py`
- 一键交付检查报告：`docs/DELIVERY_CHECK_REPORT.md`
- 一键交付检查JSON：`datasets/delivery_check_report.json`
- RoboCasa自动标注脚本：`tools/collect_robocasa_detection_dataset.py`
- RoboCasa自动标注试集：`datasets/robocasa_detection_smoke_v1/`
- RoboCasa定向数据集：`datasets/robocasa_detection_v1/`
- RoboCasa微调检测器：`models/coco_household/yolov8n_16cls_robocasa_synth_v1/weights/best.pt`
- RoboCasa第二轮困难样本：`datasets/robocasa_detection_hard_v2/`
- RoboCasa第二轮检测器：`models/coco_household/yolov8n_16cls_robocasa_synth_v2_hardneg/weights/best.pt`
- 目标实例IoU评估：`outputs/robocasa_target_detection_eval_synth_v2/target_detection_report.json`
- 演示入口：`final_demo.py`

## 十五、当前不应继续运行的训练

当前冻结测试已经100%，不应继续针对同一480条题库训练来制造更高分。下一次训练应等待新增人工盲测指令，并只针对真实错误进行。

## 最终判断

项目已经有可运行的视觉动作数据处理、特征提取和动作序列学习链路，也已建立15类任务与120项物体/设备操作语义。按图片最终确认的测试方法，覆盖15类和120项的冻结标准指令测试已完成，整序列准确率100%，通过≥90%要求。

适合对外使用的准确表述是：

> 已完成视觉动作数据处理与技能序列学习原型，建立15类任务和120项物体/设备操作语义；按标准指令映射标准动作序列的指定方法完成480条冻结基准，正式测试120/120整序列正确，达到≥90%要求。机械臂、仿真和真机执行不属于本轮必选范围。
## 2026-07-07 补充：用户截图公开数据源核验与 13 类补图状态

已按用户截图把新增公开数据源单独整理到：

- `docs/PUBLIC_DATASET_USER_SUGGESTED_SOURCES.md`
- `datasets/public_dataset_user_suggested_sources.csv`

当前判断：

1. OpenRoboCare 可作为养老陪护/护理类动作任务证据，不作为 13 类物体检测直接补图来源。
2. Oxford seven-segment / glucose & BP meters 已核验，有公开页面和下载入口，适合后续增强 `glucose_meter` 或屏幕读数能力。
3. COCO / COCO-Stuff 已经作为本地公开数据来源使用，不重复下载。
4. RoboCasa 已核验，适合家居机器人动作迁移、仿真任务和交互动作证据。
5. Roboflow 截图中的 doorbell、Extended_Devices、Windows-tio60 暂不能凭截图自动下载；需要精确 Universe 链接、版本号、API Key 或导出 ZIP。

13 类候选图片数量已经全部达到每类不少于 20 张，共 269 张。当前主要未完成项是人工筛选和框标注，不再是下载数量不足。

下一步优先级：

1. 筛选 `datasets/public_dataset_13_images/public_image_review_sheet.csv`，把合格图片标为 `keep`。
2. 生成标注模板，只让 `keep` 图片进入 YOLO 框标注。
3. 如果筛选后某类合格图少于 20，再定向补充：
   - `glucose_meter`：Oxford 专项数据；
   - `video_doorbell`：Roboflow doorbell 导出包；
   - `window_sill/window_glass/window_frame`：Roboflow Windows-tio60 或仿真家居样本。

## 2026-07-07 追加：人工直接删除文件后的同步结果

用户已经直接删除了一批不相关图片，没有在筛选表中逐行填写。已重新同步本地图片目录和筛选表：

- 当前本地候选图片：174 张
- 当前 `keep`：174 张
- 当前待筛选：0 张
- 当前 13 类均仍有候选图片，但多数类别已经低于“每类 20 张”的建议训练下限

当前每类剩余候选数量：

| 类别 | 剩余候选图 |
|---|---:|
| `balcony_door` | 7 |
| `child_clothing` | 12 |
| `glucose_meter` | 15 |
| `laminate_floor` | 18 |
| `marble_floor` | 12 |
| `screen_window` | 11 |
| `tile_floor` | 12 |
| `video_doorbell` | 11 |
| `vinyl_floor` | 14 |
| `window_frame` | 12 |
| `window_glass` | 13 |
| `window_sill` | 21 |
| `wood_floor` | 15 |

Roboflow 页面补充确认：

1. `under_surveillance` 页面中，`doorbell` 按用户确认作为 `video_doorbell` 候选；其余摄像头类不导入。
2. `window-detection-wjfqc` 页面中的图片按用户确认属于窗户、窗框、窗玻璃，可作为 `window_frame/window_glass` 候选，但若导出标签只有 `window`，仍需二次拆分。

新增导入规则文件：

- `docs/ROBOFLOW_CONFIRMED_DATASET_IMPORT_PLAN.md`

本地已经创建 Roboflow 导出包接收目录：

- `datasets/roboflow_exports/`

Roboflow 自动下载状态：

- 当前环境普通网络无法直接连接 Roboflow；
- 进一步联网访问申请被系统拒绝，原因是 Codex 当前使用额度/联网审批不可用；
- 因此本回合不能由 Codex 直接从 Roboflow 页面导出 ZIP；
- 需要用户先在浏览器页面手动导出 ZIP，并放入 `datasets/roboflow_exports/`，随后由项目脚本继续导入、去重、合并标注。

筛选表和标注模板状态：

- 已将用户删除后剩余图片全部同步为 `keep`；
- 已生成 `datasets/public_dataset_13_images/public_annotation_template.csv`；
- 仍未完成的是：每类不足 20 张的补充数据导入，以及 YOLO 框标注。

## 2026-07-08 追加：桌面 Roboflow 可视门铃/窗户数据已归类整理

用户已把 Roboflow 下载结果放到桌面：

- `C:\Users\sjtu101\Desktop\可视门铃`
- `C:\Users\sjtu101\Desktop\窗户`

已完成整理：

1. 原始导出完整备份到：
   - `datasets/roboflow_exports/doorbell_under_surveillance_v1`
   - `datasets/roboflow_exports/window_detection_wjfqc_v1`
2. 已生成清洗后的 YOLO 训练集：
   - `datasets/roboflow_imported_yolo/video_doorbell_yolo`
   - `datasets/roboflow_imported_yolo/window_coarse_yolo`
3. `video_doorbell_yolo` 只保留 Roboflow 原类别 `doorbell`，已排除摄像头类：
   - 图片：236 张
   - 标注框：245 个
4. `window_coarse_yolo` 将窗户、窗框、窗玻璃相关标注先合并为粗类 `window_coarse`：
   - 图片：2163 张
   - 标注框：5758 个
   - 注意：该数据适合先训练“窗户/窗类”粗检测；如果要严格区分 `window_frame` 和 `window_glass`，仍需要二次部件标注。
5. 已把候选图补入 13 类图片目录：
   - `video_doorbell` 当前 247 张
   - `window_frame` 当前 92 张
   - `window_glass` 当前 93 张
6. 已重新生成筛选表和标注模板：
   - `datasets/public_dataset_13_images/public_image_review_sheet.csv`
   - `datasets/public_dataset_13_images/public_annotation_template.csv`

当前 13 类候选图片总数：570 张，全部标记为 `keep`。

当前已达到每类不少于 20 张的类别：

- `video_doorbell`
- `window_frame`
- `window_glass`
- `window_sill`

仍低于每类 20 张、下一步需要补下载的类别：

| 类别 | 当前数量 | 至少还需 |
|---|---:|---:|
| `balcony_door` | 7 | 13 |
| `child_clothing` | 12 | 8 |
| `glucose_meter` | 15 | 5 |
| `laminate_floor` | 18 | 2 |
| `marble_floor` | 12 | 8 |
| `screen_window` | 12 | 8 |
| `tile_floor` | 12 | 8 |
| `vinyl_floor` | 14 | 6 |
| `wood_floor` | 15 | 5 |

桌面导入报告：

- `docs/ROBOFLOW_DESKTOP_IMPORT_REPORT.md`
- `datasets/roboflow_desktop_import_report.json`
- `datasets/roboflow_desktop_import_manifest.csv`

## 2026-07-08 追加：阳台门/阳台窗改为阳台栏杆，并扩大训练图片规模

用户确认：阳台窗户/阳台门不方便查找，且容易与普通房间窗户混淆，因此该类改为：

- 旧：`balcony_door`
- 新：`balcony_railing`

已完成：

1. 桌面 Roboflow 阳台数据已整理为 `balcony_railing`。
2. 旧 `balcony_door` 已归档，不再进入当前训练集。
3. 已修正相关工具，后续统计和下载计划按 `balcony_railing` 计算。
4. 已用本地公开数据缓存把多类型地板和儿童服装等类别扩充到 100 张级别。
5. 已尝试联网补下载，但当前环境网络权限不稳定；Wikimedia Commons 只补到少量 `glucose_meter` / `child_clothing`，Roboflow/Kaggle 仍建议手动导出 ZIP。

当前按每类 100 张目标统计：

| 类别 | keep 数 | 状态 |
|---|---:|---|
| `balcony_railing` | 100 | 达标 |
| `child_clothing` | 100 | 达标 |
| `glucose_meter` | 17 | 不达标，且仍需排除试纸/图表/读数类干扰 |
| `laminate_floor` | 100 | 达标 |
| `marble_floor` | 100 | 达标 |
| `screen_window` | 12 | 不达标 |
| `tile_floor` | 100 | 达标 |
| `video_doorbell` | 247 | 达标 |
| `vinyl_floor` | 100 | 达标 |
| `window_frame` | 100 | 达标 |
| `window_glass` | 100 | 达标 |
| `window_sill` | 21 | 不达标 |
| `wood_floor` | 100 | 达标 |

下一步只需要重点补：

1. `screen_window`
2. `glucose_meter`
3. `window_sill`

详细下载计划见：

- `docs/NEXT_DATA_DOWNLOAD_PLAN_AFTER_BALCONY_RAILING.md`

## 2026-07-08 追加：公开网站补下载 screen_window/window_sill/glucose_meter

已按用户要求从公开网站继续寻找和下载：

1. `screen_window`
   - 找到 HuggingFace `mlnomad/imnet1k_window_screen`
   - 已下载 Parquet 图片数据并抽取 300 张
   - 当前 `screen_window` keep 数：312
2. `window_sill`
   - 找到 HuggingFace `morimar/windowsill`，但其为 duckdb 数据库，不是直接图片检测集
   - 改用已下载 Roboflow 窗户数据抽取 200 张窗台二次标注候选
   - 当前 `window_sill` keep 数：221
3. `glucose_meter`
   - HuggingFace 未找到合适数据集
   - Wikimedia Commons 只能补到约 20 张，且有试纸/图表/读数干扰
   - Oxford 公开链接返回 HTML，未取得有效图片 ZIP
   - Roboflow 搜索页脚本访问 403，需要浏览器手动导出
   - 当前 `glucose_meter` clean keep 数：17，仍未达标

当前按每类 100 张目标：13 类中 12 类达标，只剩 `glucose_meter` 需要继续补。

详细记录：

- `docs/PUBLIC_WEB_DOWNLOAD_ATTEMPT_2026-07-08.md`

## 2026-07-08 追加：glucose_meter 改由 spectacles（眼镜）替代

用户确认 `remote_control` 已在“物品取送”中出现，不能作为助老服务缺口的替代项；随后指定改为“眼镜”。项目正式采用内部标准对象名：

- 旧缺口：`glucose_meter`
- 拒绝的中间替代：`remote_control`（原因：与物品取送任务重复）
- 新替代：`spectacles`

已完成：

1. 已读取用户提供的本地数据集：`C:\Users\sjtu101\Downloads\af082-main.zip`。
2. 已解压并整理 `af082-main` 眼镜照片数据集。
3. 已排除 `隐形眼镜`，因为它不是可抓取/可递送的实体眼镜。
4. 已导入 578 张 `spectacles` 候选图到 `datasets/public_dataset_13_images/spectacles/`。
5. 已生成导入清单：`datasets/spectacles_import_manifest.csv`。
6. 已生成替代报告：`docs/SPECTACLES_REPLACES_GLUCOSE_METER_REPORT.md`。
7. 已更新训练准备度检查工具，当前 13 类按 `spectacles` 统计，不再把 `glucose_meter` 作为未达标缺口。

当前按每类 100 张目标，13 类公开图片候选集已全部达标：

| 类别 | keep 数 | 状态 |
|---|---:|---|
| `balcony_railing` | 100 | 达标 |
| `child_clothing` | 113 | 达标 |
| `laminate_floor` | 100 | 达标 |
| `marble_floor` | 100 | 达标 |
| `screen_window` | 312 | 达标 |
| `spectacles` | 578 | 达标 |
| `tile_floor` | 100 | 达标 |
| `video_doorbell` | 247 | 达标 |
| `vinyl_floor` | 100 | 达标 |
| `window_frame` | 100 | 达标 |
| `window_glass` | 100 | 达标 |
| `window_sill` | 221 | 达标 |
| `wood_floor` | 100 | 达标 |

仍未完成：

1. `spectacles` 数据集是图片分类集，不是检测框数据集；需要人工框选眼镜位置后才能真正训练检测模型。
2. `screen_window`、`window_sill`、地板类等候选图也仍需按模板做框选/区域标注。
3. 完成标注后，才能进入 13 类增量 YOLO 训练和统一视觉模型评估。

## 2026-07-17 追加：自然语言同义指令与受约束动作规划

已完成“自由表达不必等于预设句子”的动作解析增强，并将模型生成结果接入动作约束层：

1. 新增倒水同义表达：`给我倒杯水`、`帮我倒水`、`倒点水过来`、`我想喝水`，统一解析为 `food_serving / water_cup / pour_water`。
2. 新增阅读灯开关短语，明确区分 `turn_on` 与 `turn_off`，避免“关闭阅读灯”生成打开动作。
3. 新增 `tools/action_constraints.py`：模型动作序列与解析出的任务、物体和操作不一致时，不直接判为通过，而是记录违规原因并采用受约束标准动作序列。
4. 手动输入测试入口保留“模型原始动作序列”，并新增“最终采用动作序列”和“动作约束纠正原因”。因此可以区分模型本身是否出错，以及系统最终是否给出了安全、正确的步骤。
5. 新增 10 条回归盲测，覆盖阅读灯开关、电视开关、浇花/养护花及四种倒水自然表达。当前解析准确率和最终动作序列准确率均为 100%。

当前边界：该 100% 是这 10 条固定回归样例的结果，不代表所有开放式中文表达都已达到 100%。后续应持续加入人工新写的同义句和容易混淆的反向指令，作为独立盲测数据。

## 2026-07-17 追加：414 条自然语言盲测与独立复测

当前正式目录以 **138 条任务—物体操作关系** 为评测场景；其中有 123 个不重复物体。同一物体出现在不同任务时，期望动作可能不同，因此不能只按物体数量测试。

已新增独立盲测流程：每条关系由独立人员填写 3 条自然语言表达，共 **414 条**。空白填写表为：

- `datasets/instruction_blind_paraphrase_template_414.csv`

填写时只显示任务场景、物体和目标位置，**不显示标准动作序列**。该表后来已由用户填写 173 条、辅助工具补写 241 条，因而只作为混合预评测诊断，不能将当前标准基准或 10 条回归样例表述为“原始模型对开放式语言已经超过 90%”。

填写完成后依次运行：

```powershell
.\.venv\Scripts\python.exe prepare_blind_paraphrase_test.py --mode build-test
.\.venv\Scripts\python.exe evaluate_independent_blind_action_test.py --fail-below-raw-threshold
```

评测报告会严格给出两项互不替代的指标：

1. **原始模型整序列准确率** `raw_exact_sequence_accuracy`：记录自然语言解析和模型生成后的原始动作，不执行动作约束。只有这项在冻结 414 条盲测上达到 90%，才能称“原始模型动作解析率超过 90%”。
2. **最终系统整序列准确率** `final_exact_sequence_accuracy`：在保留原始动作后，再执行安全动作约束得到的最终结果。它反映可交付系统效果，不能替代原始模型分数。

报告还会保存逐条原始动作、最终动作、纠正情况和 95% Wilson 置信区间，便于中期检查复核。

### 2026-07-17 首次混合预评测结果

用户已人工填写 173 条，其余 241 条由辅助工具按日常表达补写；因此该版本是“混合人工/辅助预评测”，不是可对外宣称泛化能力的独立人工盲测。

在冻结的 414 条混合预评测上，以 `beam-width=1` 运行得到：

- 原始模型整序列准确率：**65.22%**，95% Wilson 区间为 60.51%–69.65%。
- 最终系统整序列准确率：**65.70%**，95% Wilson 区间为 61.00%–70.11%。
- 指令解析准确率：**73.19%**。

因此当前真实结论是：该自然语言预评测**未达到 90%**，不能写成“开放式指令动作解析率超过 90%”。

已修复一个影响最终系统评分的兼容问题：合并后的 `cleaning` 任务原先仍走旧任务名查表，导致约束层把部分正确动作清空；修复后最终系统从 21.50% 恢复到 65.70%。其余主要问题是自然表达任务歧义、旧解析词表缺少部分对象，以及娱乐服务、养护管理、衣物清洗等任务的动作/对象区分不足。

预评测报告：

- `datasets/independent_blind_action_test_414_beam1_report.json`
- `docs/INDEPENDENT_BLIND_ACTION_EVALUATION_BEAM1_REPORT.md`

### 下一次正式复测（仅人工填写）

为避免把已经见过的 414 条混合预评测句子再次当作考题，已生成新的空白复测表：

- `datasets/human_only_instruction_retest_template_414.csv`

它仍覆盖当前 **138 条任务—物体操作关系**，每条关系 3 个表达，共 414 行；其中 `human_instruction` 全部为空，且不包含标准动作序列。必须由未查看标准答案的人工填写者填写，填写完成后才可冻结并评估。届时报告将继续分别给出原始模型分数与最终系统分数；只有新的全人工冻结集上的原始模型整序列准确率达到至少 90%，才能宣称自然语言动作解析达到该指标。

## 2026-07-20 追加：自然语言结果展示

弹窗和默认文字报告现保留用户输入的原话，并用中文说明“我理解为”和“建议步骤”；内部任务、物体、动作代码移动到“技术详情（供检查）”部分。**自然语言展示不改变解析、原始模型分数或最终系统分数**，仅让用户更直观地确认系统是否理解了日常说法。

## 2026-07-20 追加：混合语义动作解析正式终测（一次性）

为提升自然语言动作解析，项目新增“可训练的文本关系分类器 + 标准动作规划器”的混合方案：

1. 从当前 138 条“任务—物体操作关系”生成 5,382 条训练表达和 138 条开发表达；每条生成表达都明确包含操作含义。
2. 训练字符级 n-gram 文本分类器预测“任务—物体关系”，再由该关系检索当前目录中对应的标准动作序列。
3. 开发集结果为 138/138；这是生成表达上的工程验证，**不能代替人工终测**。
4. 训练数据生成前，已把用户填写的 414 条人工复测指令转换为不可逆 SHA-256 哈希；训练和开发过程未读取这些终测文本或动作答案。

模型与目录固定后，已对冻结的 414 条人工指令运行**唯一一次**正式终测，并生成不可重复运行锁。结果如下：

| 指标 | 结果 | 是否达到 90% |
|---|---:|---|
| 原始混合系统整序列准确率 | **21.26%（88/414）** | 未达到 |
| 约束后整序列准确率 | **21.26%（88/414）** | 未达到 |
| 指令关系解析准确率 | **21.26%（88/414）** | 未达到 |

因此，当前项目不能宣称“开放式自然语言动作解析率超过 90%”。本轮人工终测已经解封，**不得**再将其用于训练、阈值调参或重复评测。若后续要继续提高准确率，只能使用不含这 414 条内容的新训练/开发数据完成改进，再由独立人工填写一份全新的冻结终测表评估。

相关文件：

- `datasets/human_only_final_instruction_hashes.json`：414 条终测指令的哈希隔离清单，不含原文。
- `models/semantic_action/dev_report.json`：训练阶段的开发集记录。
- `models/semantic_action/development_report.json`：混合系统开发评测记录。
- `models/semantic_action/final_human_414_report.json`：一次性正式终测报告。
- `models/semantic_action/final_evaluation_lock.json`：终测锁及模型、测试集、报告哈希。

## 2026-07-20 追加：扩充样本训练与助手生成自测

在不读取、训练、调参或重跑已锁定 414 条人工终测的前提下，项目新增了一轮隔离的日常语言扩充：

1. 为全部 138 条任务—物体关系生成了 **4,416 条训练表达**和 **1,104 条开发表达**，覆盖口语请求、时间/原因语境、语序变化和同义动词；每条表达都有明确操作意图。
2. 新训练集、开发集与旧人工终测均通过 SHA-256 哈希查重；旧人工终测仅使用其哈希清单，不读取原文或答案。
3. 新模型写入独立目录 models/semantic_action_augmented/，没有覆盖上一轮模型或一次性终测记录。
4. 在 1,104 条新增开发表达上，关系识别、原始整序列和约束后整序列准确率均为 **99.82%（1102/1104）**。两条开发误差均涉及 balcony_railing（阳台栏杆）表达。
5. 模型固定后，助手生成并冻结了 138 条自测指令（每个关系一条），并与训练、开发和旧人工终测哈希隔离。

### 助手生成自测结果

| 指标 | 结果 |
|---|---:|
| 关系识别准确率 | **98.55%（136/138）** |
| 原始整序列准确率 | **98.55%（136/138）** |
| 约束后整序列准确率 | **98.55%（136/138）** |

两条自测失败均为清洁任务中 cup、mug 与 bowl 的相近容器类别混淆；未将这些自测答案回灌训练。

**证据边界：** 该 98.55% 是“助手生成的开发自测”结果，不是独立人工盲测，不能覆盖或推翻前述 414 条一次性人工终测的 21.26% 结果，更不能据此宣称真实自然语言动作解析已经超过 90%。若需恢复“人工 >90%”的目标，仍必须在后续模型完全固定后，由独立人员填写新的、未见过答案的人工测试集。

相关文件：

- datasets/semantic_action_augmented_train.json
- datasets/semantic_action_augmented_dev.json
- datasets/assistant_semantic_selftest_138.json
- models/semantic_action_augmented/development_report.json
- models/semantic_action_augmented/assistant_selftest_report.json
- docs/ASSISTANT_SEMANTIC_SELFTEST_REPORT.md

## 2026-07-20 追加：固定标准指令正式验收

为对应“将系统生成的动作序列与预设标准动作序列进行对比、准确率不低于 90%”的验收要求，项目已建立并冻结正式固定指令验收集：

- 覆盖范围：当前 **138 条任务—物体操作关系**；
- 每条关系：1 条标准指令 + 3 条固定常见同义/口语指令；
- 验收总数：**552 条**；
- 评分规则：动作名称、参数和顺序与预设标准动作序列完全一致才算通过；
- 90% 门槛：至少 **497/552** 条原始动作序列通过。

正式评测使用扩充后的语义分类模型，先记录模型的原始动作序列，再额外记录约束后序列。正式成绩只取原始整序列结果：

| 指标 | 结果 |
|---|---:|
| 原始动作整序列通过数 | **540/552** |
| 原始动作整序列准确率 | **97.83%** |
| 90% 门槛 | 497/552 |
| 正式验收结论 | **通过** |

本验收集已与增强训练集、增强开发集、助手自测集及已锁定人工终测哈希清单进行查重，且没有用于训练或开发。第三方可使用下列命令复跑：

~~~powershell
.\.venv\Scripts\python.exe evaluate_official_fixed_command_acceptance.py
~~~

复核文件：

- datasets/official_fixed_command_acceptance_552.json：552 条冻结指令、对应标准动作和逐条验收目标；
- datasets/official_fixed_command_acceptance_hashes.json：验收指令哈希和目录哈希；
- models/semantic_action_augmented/official_fixed_command_acceptance_report.json：逐条原始输出、标准动作、通过状态及模型/数据/目录哈希；
- docs/OFFICIAL_FIXED_COMMAND_ACCEPTANCE_REPORT.md：便于阅读的正式结论。

**适用范围：** 97.83% 证明系统在上述冻结的固定标准指令及其固定常见表达范围内，能够把指令正确映射为规定动作序列，满足该动作解析验收要求；它不等同于任意无限开放中文表达都达到 97.83%。
