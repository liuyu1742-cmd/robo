# 基于视觉的人类动作捕获、解析和技能学习迁移技术中期检查说明

生成时间：2026-07-16 17:27

## 1. 项目目标与当前交付口径

本项目中期检查按两个主线组织：一是操作技能学习，即围绕 15 种家庭/助老典型任务和 120 个物体建立可识别、可解析、可测试的技能序列；二是动作解析，即把用户指令或视频片段转成标准动作序列，并用预设标准序列进行自动评分。

当前阶段已经完成 15+120 标准清单、动作序列模板、指令到动作序列测试入口、EPIC-KITCHENS 视频片段特征提取，以及多来源图片/视频/标注证据整理。需要特别说明：120个物体是技能学习和动作解析的覆盖清单，不等于每个物体都已经完成独立视觉检测模型训练。

## 2. 方法来源：开源底座 + 本项目适配修正

本项目不是直接把某一个 GitHub 现成项目下载后作为最终成果，也不是完全从零发明底层视觉算法。更准确的说法是：底层视觉识别、视频特征提取和模型训练使用了公开数据集与开源深度学习工具作为基础；项目自己的主要工作是把这些通用资源改造成符合“15种任务、120种物体、动作解析准确率测试”要求的一套完整流程。

公开/开源底座主要包括：PyTorch/torchvision 训练框架，YOLO 系列目标检测思路，R3D-18 视频特征骨干，FFmpeg/imageio 视频切片工具，以及 COCO/LVIS、HICO-DET、HOI4D、EPIC-KITCHENS、RoboCasa、Roboflow/YCB 等公开或可整理的数据来源。这些资源提供的是“看图/看视频/训练模型”的基础能力和原始样本。

本项目的中心修正和整合工作包括：第一，建立符合任务书的 15 种家庭/助老任务和 120 个可操作物体清单；第二，把公开数据集中的类别、视频片段、检测框和动作事件映射到本项目物体，例如把摄像头、门铃按钮、阳台栏杆、窗框、地板材质等整理到对应任务目录；第三，为每个物体生成标准高层动作序列，例如 locate、grasp、move、wash、wipe、inspect；第四，增加自然语言同义指令入口，使用户输入“清洗衣物”“洗衣服”“把衣服洗一下”等不同说法时能归到同一任务；第五，建立自动评分程序，用标准动作序列评估系统输出的完整匹配率、步骤覆盖率和顺序正确率。

因此，中期文档中的“方法”应理解为一种项目化集成方法：用开源模型和公开数据解决视觉输入问题，用本项目定义的任务清单和动作模板解决技能学习目标，用评测脚本把模型输出和标准答案进行对比。开源部分提供底座，本项目修正部分提供任务口径、数据映射、动作语义和验收指标。

## 3. 技能学习使用的数据集、程序和测试结果

- 结构化技能数据：`datasets/standard_instruction_action_benchmark.json`，覆盖 15 种任务、120 个物体，每个物体配置标准动作序列，用于测试“给定指令后能否输出正确动作步骤”。
- 项目动作训练/验证/测试集：共 20997 条样本，训练集 15524 条，验证集 2313 条，测试集 3160 条。
- 动作数据来源：EPIC-KITCHENS 5674 条、HICO-DET 12350 条、HOI4D 2973 条。
- 视觉/物体证据数据：COCO/LVIS、OpenImages 派生 YOLO 集、HICO-DET、HOI4D、RoboCasa、YCB、EPIC-KITCHENS、Roboflow 和公开补充图片。它们的作用是为部分物体提供图片、检测框、视频片段或动作事件证据。
- 主要程序：`train_project_transformer_v2.py`、`train_instruction_action_transformer.py`、`evaluate_project_model_v2.py`、`manual_instruction_entry.py`、`run_acceptance_check.py`。

典型动作包括 locate、grasp、move、place、release、wash、rinse、wipe、sweep、mop、inspect、operate、cook、prepare 等。它们不是机械臂底层控制指令，而是中期检查口径下的高层技能步骤。

120个物体的训练情况需要分两层理解。第一层是“技能/动作序列模型”：120个物体都进入了标准任务清单和动作序列模板，模型学习的是在某个任务下应输出哪些步骤，例如 locate(cup) → grasp(cup) → move(sink) → wash(cup)。这一层已经可以用标准序列准确率、步骤覆盖率等指标评估。第二层是“视觉检测模型”：它负责从图片或视频中识别物体，目前只对已有公开视觉数据和标注的物体形成了较强证据，并没有宣称120个物体都已完成检测器训练。

现有视觉检测映射中，已有检测类别可关联到约 42 个项目物体；其余物体主要依靠动作模板、动作记录或后续补充数据。该模型的用途是给动作解析提供视觉条件，例如图中检测到 cup、sink 后，系统更容易判断应进入“餐具清洗/取送水杯”等动作序列。

技能学习与动作序列测试结果：
- 标准 15+120 指令动作基准：100.00% 严格序列准确率。
- 项目归一化测试集：99.37% 序列准确率。
- 视觉条件动作模型测试集：99.84% 序列准确率。
- 总验收检查：通过。

## 4. 动作解析使用的数据集、程序和测试结果

- 视频来源：EPIC-KITCHENS Tier A 子集，已下载并切分 480p 片段。
- 视觉时序特征：R3D-18 视频骨干，16 帧/片段，输出 512 维时序特征。
- 物体检测特征：YOLO 家庭物体检测模型，按片段抽帧提取空间/物体存在特征。
- 融合程序：`train_epic_temporal_conditions.py`、`train_epic_visual_conditions.py`、`train_epic_fused_conditions.py`、`process_epic_tiered_clips.py`。
- 已处理片段数：5674。
- EPIC 融合模型验证集最佳轮次：5。
- 早期 EPIC 视觉条件模型 joint accuracy：2.00%。

动作解析的测试方法是：系统先根据输入指令或视频/视觉特征生成动作序列；测试阶段再把生成序列与预先定义的标准动作序列对比，计算完整匹配、步骤覆盖率、顺序覆盖率、动作词准确率和参数准确率。标准序列用于评分，不是让模型在输出时直接读取答案。

## 5. 方法原理与为什么可以给出准确率

代码采用“公开视觉/视频特征 + 项目任务清单 + 动作序列模板 + Transformer/分类模型 + 自动评分”的路线。先把家庭任务拆成可比较的动作 token，例如 locate(cup)、grasp(cup)、move(sink)、wash(cup)。训练时模型学习从指令、任务、物体和视觉条件映射到动作序列；测试时只给输入，不给标准答案。模型输出后，再由评测程序和标准序列逐步比较，因此可以得到严格序列准确率、步骤覆盖率、编辑相似度等指标。

从原理上看，系统分为三层：第一层是视觉证据层，把图片、视频片段和检测框整理成物体或场景证据；第二层是语义动作层，把“清洗衣物、清洁地面、检查门铃”等自然语言任务转成统一的任务ID、物体ID和动作 token；第三层是评测层，把系统生成的动作序列与预设标准序列比较。这样做的好处是，即使暂时不接入机械臂，也能客观评估系统是否理解了任务、找到了对象、输出了合理步骤。

需要说明的是，当前准确率主要评价“动作解析/技能序列生成”的正确性，不等于所有120个物体的视觉检测准确率都已经达到同一水平。视觉检测部分仍受数据量和标注质量限制；项目目前通过公开数据整理、弱标注和部分真实标注逐步补齐。

当前高准确率主要来自结构化 15+120 标准动作模板和归一化动作序列。它证明“指令到标准动作步骤”的解析链路已经可用；但视觉检测模型并未对120个物体全部完成同等强度训练，因此不能把动作解析准确率直接等同于120类物体视觉识别准确率。

## 6. 15 种典型任务与 120 个物体清单

### 清洁（cleaning）

`carpet`、`wood_floor`、`tile_floor`、`marble_floor`、`laminate_floor`、`vinyl_floor`、`robot_vacuum`、`cup`、`mug`、`bowl`、`plate`、`spoon`、`fork`、`knife`、`pot`、`toilet`、`sink`、`bathtub`、`shower`、`mirror`、`bathroom_floor`、`faucet`、`soap`

### 整理任务（organizing）

`storage_box`、`bookshelf`、`wardrobe`、`drawer`、`cabinet`、`desk`、`coffee_table`、`toy`

### 智能烹饪（smart_cooking）

`rice_cooker`、`induction_cooker`、`oven`、`microwave`、`frying_pan`、`electric_kettle`、`blender`、`pressure_cooker`

### 家电综合管理（appliance_management）

`ceiling_light`、`reading_lamp`、`television`、`air_conditioner`、`refrigerator`、`washing_machine`、`electric_curtain`、`fan`、`remote_control`

### 智慧安防（security_monitoring）

`smart_lock`、`security_camera`、`fire_alarm`、`fire_extinguisher`、`doorknob`、`padlock`、`doorbell_button`、`video_doorbell`

### 衣物清洗（laundry）

`child_clothing`、`shoes`、`cotton_coat`、`down_jacket`、`wool_garment`、`shirt`、`pants`、`towel`

### 垃圾处理（waste_disposal）

`trash_bin`、`recyclable_bottle`、`food`、`plastic_bag`、`cardboard_box`、`aluminum_can`、`glass_bottle`、`battery`

### 服装护理（clothing_care）

`dress`、`suit`、`sweater`、`skirt`、`socks`、`scarf`、`bed_sheet`、`iron`

### 门窗护理（window_care）

`window`、`window_glass`、`window_frame`、`curtain`、`blind`、`screen_window`、`balcony_railing`、`window_sill`

### 卧室服务（bedroom_service）

`bed`、`pillow`、`quilt`、`mattress`、`nightstand`、`bedroom_lamp`、`clothes_hanger`、`laundry_basket`

### 送餐饮水（food_serving）

`water_cup`、`wine_glass`、`bottle`、`food_tray`、`dinner_plate`、`serving_bowl`、`chopsticks`、`thermos`

### 物品取送（object_fetching）

`remote_control`、`mobile_phone`、`laptop`、`book`、`spectacles`、`walking_cane`、`keys`、`wallet`

### 医疗服务（elderly_assistance）

`blood_pressure_monitor`、`thermometer`、`bandage`、`first_aid_kit`、`wheelchair`、`sanitary_pad`、`medical_mask`、`medical_glove`

### 维护管理（maintenance_management）

`robot_vacuum`、`fire_alarm`、`washing_machine`、`air_conditioner`、`refrigerator`、`fan`、`electric_curtain`、`smart_lock`、`flower_pot`、`watering_can`

### 娱乐服务（entertainment_service）

`remote_control`、`television`、`laptop`、`book`、`mobile_phone`、`toy`、`game_controller`、`speaker`

## 7. 当前全项目数据整理结果

- 全项目按120物体整理后，有图片证据的物体数：120。
- 全项目按120物体整理后，有视频证据的物体数：36。
- 全项目按120物体整理后，有标注或动作记录的物体数：120。
- 全项目已整理图片数：99117。
- 全项目已整理视频数：3020。
- 全项目已整理标注/动作记录数：186923。
- 已生成弱标注/预标注的物体数：110。
- 已生成或可用的弱标注文件数：78611。
- 暂无图片、无法凭空标注的物体数：10。
- 没有任何图片/视频/标注或动作记录的物体数：0。
- 没有图片或视频视觉样本的物体数：0。

这批数据已经按 15 任务目录移动整理，交付目录为：`datasets/midterm_15task_delivery/`。所谓“已有数据证据的物体”，是指该物体目录下已经存在图片、视频、检测标签、动作记录或来源元数据中的至少一种。

标注口径：已有 YOLO/HICO/RoboCasa/Roboflow 标注的样本已移动原始标注；EPIC/HICO/HOI4D 的动作样本已生成 action_records 动作标注；对已有图片但缺少检测框的数据，本次新增 YOLO 格式弱标注/预标注，位置在各物体目录的 `annotations/weak_yolo_labels/` 下。弱标注采用全图框，作用是让图片先具备可训练草稿标签，后续仍需人工复核，不伪造成已人工精确标注。

为什么仍有不少物体没有图片或视频：第一，120个物体是为了覆盖任务技能而设计的完整清单，其中有些物体在公开数据集中很少见，例如 smart_scale、bandage、walking_cane、glucose_meter 等；第二，COCO/HICO/HOI4D/EPIC 等公开数据集本来不是按本项目120物体设计的，很多类别名称不能一一对应；第三，有些物体只有动作模板或动作记录，没有可直接移动的原始图片/视频；第四，为避免误标，本次只移动能明确对应的证据，无法确认的样本没有强行归类。

## 8. 未完成任务与风险

1. 部分新增图片目前仍是候选图或分类图，需人工框选/区域标注后才能进入视觉检测训练。
2. 120 个物体中并非每个物体都有独立图片、视频和标注；本次交付目录已把缺口显式标出，不能把没有视觉样本的物体说成已经完成视觉训练。
3. EPIC-KITCHENS 视频解析验证已完成特征抽取和初步训练，但视频端到端识别多动作、多物体的准确率仍低于结构化指令动作解析。
4. 当前不接入机械臂底层控制，因此“完成动作”指的是输出标准动作序列并通过规则/模型评测，不代表真实机器人执行成功。
5. 若后续要把视觉识别能力纳入最终准确率，需要继续补齐缺少视觉数据的物体、完成人工标注或高质量预标注，并按独立测试集报告 mAP、召回率、误检率。
