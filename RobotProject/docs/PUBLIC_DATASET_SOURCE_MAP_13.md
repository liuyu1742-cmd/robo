# 13类缺口公开数据源映射

> 本文件按“不自采，只从公开网站/公开数据集查找”的要求更新。  
> 这是公开来源映射，不代表图片已经下载、标注或训练完成。

## 来源依据

- Open Images V7：大规模公开图像标注，含框、分割、关系、点级和图像级标签。
- COCO/COCO-Stuff：COCO-Stuff给COCO图像增加像素级stuff标注，适合地面、墙面、门窗等区域。
- ADE20K：场景解析数据，含物体和部件标注，适合窗户部件风险处理。
- Fashionpedia：服装、服装部件、属性和分割标注。
- Wikimedia Commons：自由媒体库，适合查找门铃、血糖仪、儿童服装等具体对象候选，但要逐文件检查许可证。

## 13类映射表

| 对象 | 推荐公开来源 | 可用标签/类别 | 当前判断 | 风险 |
|---|---|---|---|---|
| `wood_floor` | COCO-Stuff | `floor-wood` | 可作为公开数据证据 | 材质粗分类，不等于所有木地板都能精确识别 |
| `tile_floor` | COCO-Stuff | `floor-tile` | 可作为公开数据证据 | 瓷砖/石材/大理石边界可能混淆 |
| `marble_floor` | COCO-Stuff | `floor-marble` | 可作为公开数据证据 | 与普通石材/瓷砖可能混淆 |
| `laminate_floor` | COCO-Stuff + 材质属性说明 | `floor-wood`子属性 | 只能作为弱证据 | 不能直接宣称有独立公开标签 |
| `vinyl_floor` | COCO-Stuff + 材质属性说明 | `floor-other`或材质属性 | 只能作为弱证据 | 不能直接宣称有独立公开标签 |
| `video_doorbell` | Wikimedia Commons + Open Images候选 | Doorbells / doorbell buttons / electric doorbells | 可作为公开媒体候选 | 普通门铃不等于可视门铃，需人工筛选摄像头/视频门铃外观 |
| `child_clothing` | Fashionpedia + Wikimedia Commons | clothing / apparel parts / children's clothing | 可作为组合证据 | 必须保留“儿童属性”人工映射 |
| `window_glass` | ADE20K + COCO-Stuff | window parts / `window-other` | 可作为结构证据 | 不是独立物体，应做部件/分割 |
| `window_frame` | ADE20K + COCO-Stuff | window parts / `window-other` | 可作为结构证据 | 框与墙/窗边界依赖分割口径 |
| `screen_window` | Wikimedia Commons + ADE20K候选 | screen/window mesh候选 | 弱证据 | 纱窗网格需要人工筛选 |
| `balcony_door` | ADE20K + COCO-Stuff | door/window structure / `door-stuff` | 可作为结构证据 | 阳台门和普通门/窗需要场景属性区分 |
| `window_sill` | ADE20K + COCO-Stuff | window parts / structural region | 弱证据 | 窗台通常是结构部件，不是独立物体 |
| `glucose_meter` | Wikimedia Commons | Glucose meters / electronic glucose meters | 可作为公开媒体候选 | 医疗设备外观多样，需排除试纸/图表/历史仪器 |

## 前四步执行结果

1. 已把两个任务拆开：任务A为动作解析验收，任务B为公开数据集/代码整理。
2. 已为13类缺口建立公开来源映射表。
3. 已把图片中三条风险写入风险登记表。
4. 已更新总状态文档，后续交付按任务A/任务B分开汇报。
