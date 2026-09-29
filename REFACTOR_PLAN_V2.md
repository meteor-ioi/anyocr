# AnyOCR 原生 Rust 引擎版面与表格双轨重构实施方案

> **文档定位**：基于 SensiDoc 中已实测验证的端侧极轻量版面与表格识别双轨流水线（PicoDet-S + 全套 PP-OCRv6 Det/Rec ONNX + SLANet_plus），分析并设计对现有原生 Rust 项目 [`anyocr`](file:///Users/icychick/Projects/anyocr/) 的重构路线图与实施指南。

---

## 一、 现状诊断：AnyOCR 排版撕裂的深层根因

在重构前审视现有 `anyocr` 的架构实现，发现了导致其历史排版撕裂的关键短板：

```plaintext
[当前 AnyOCR 的单轨困局]
整页图像 ───┬───> PP-OCRv6 Det/Rec ───> 文本行与坐标
            │
            └───> SLANet (全图强切) ───> 强行将整页当表格预测！
                                             │
                                             ▼
                                  页眉/段落被切成碎格
                                             │
                                             ▼
                 不得不写上千行启发式规则修补 (grid_table.rs / table_matcher.rs)
```

1. **缺失版面分析（Layout Detection）前置分流**：
   * `anyocr` 的 `engine.rs` 中，直接将整张原图送入 `table_predictor`（SLANet）；
   * SLANet 专为“单个裁剪好的表格”设计，整页输入会产生灾难性的全图单元格误召回，将标题、公司名、地址切成大量畸形单元格；
2. **规则库过度膨胀与脆弱性**：
   * 为修补全图强切的副作用，现有代码堆砌了 `core_table_row_range`、虚线去重、网格合并等大量硬编码启发式规则，代码量超 3,000 行，但面对真实工业单据仍极易失效；
3. **识别词典未对齐**：
   * 原生识别模块若未精确绑定完整的 18,708 字符 `ppocrv6_dict.txt`，英文单词之间便会发生严重粘连。

---

## 二、 重构核心思路：引入双轨解耦流水线

将 SensiDoc 验证成功的“双轨协同流水线”原生迁移至 AnyOCR 的 Rust 体系中：

```plaintext
[重构后的 AnyOCR 双轨原生流水线]
                     [ 原图 / PDF 光栅化图像 ]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [ 轨一：文本与字符基线 ]          [ 轨二：版面与区域探测 ]
       PP-OCRv6 Det (ONNX)               PicoDet-S Layout (4.9MB ONNX)
                 │                               │
                 ▼                               ▼
       PP-OCRv6 Rec (ONNX)               划分语义区块 (Table / Text / Seal)
       (18,708 字符专属字典)                      │
                 │                               ├─ Non-Table 区域 (Text/Title)
                 │                               │  直接按阅读顺序保留为正文/段落
                 │                               │
                 │                               ▼
                 │                       Table 区域 (局部 ROI 裁剪)
                 │                               │
                 │                               ▼
                 │                       SLANet+ (7.6MB ONNX)
                 │                       仅对局部 ROI 预测表格骨架与单元格
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     [ 空间投影与内容填充引擎 ]
                     (单元格 BBox 与文本行交并映射)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     [ 高保真 GFM Markdown 表格 ]       [ 物理单元格绝对点坐标索引 ]
     (彻底消除碎表格与错位)            (提供给 SensiDoc 精准原件高亮与脱敏)
```

---

## 三、 重构可行性全方位分析

### 1. 运行时与依赖完全匹配 (100% 原生可行)
* **纯 ONNX Runtime**：`anyocr` 已经完全基于 Rust `ort` (2.0.0-rc.9) 构建，免除任何 Python 依赖；
* **现有模型完备**：`anyocr/models/ocr/` 目录下已经存在 `PP-OCRv6_det_small.onnx`、`PP-OCRv6_rec_small.onnx`、`ppocrv6_dict.txt`、`slanet-plus.onnx`；
* **仅需新增一个超轻量权重**：仅需引入 `picodet_s_layout_17cls.onnx`（仅 **4.9 MB**），即可补齐版面检测这一核心缺失环节！

### 2. 性能与资源预期
* **推理耗时**：
  * Python 下 47 页实测单页耗时 1.08 秒；
  * Rust 原生 `ort` 结合连续内存切片与多线程调度，单页预计可压至 **0.5 ~ 0.8 秒**；
* **内存占用**：
  * 摆脱 Python 解释器与 Paddle C++ 框架层开销，纯 Rust 进程的常驻内存预计可从 1.4GB 进一步压减至 **300MB ~ 600MB** 以内。

---

## 四、 详细实施阶段与演进路线

### 第一阶段：引入 `LayoutDetector` 原生推理模块
1. **模型导出与落盘**：
   * 将 `PicoDet-S_layout_17cls` 转换为标准 ONNX 格式（4.9MB），放置于 `models/ocr/picodet_s_layout.onnx`；
2. **新增检测器**：
   * 在 `src/models/` 下新建 `layout_detector.rs`；
   * 实现输入尺寸规整（如 800x800）、非极大值抑制（NMS）、多标签分类（`table`, `title`, `text`, `figure`, `seal` 等）与坐标逆变换。

### 第二阶段：重构 `engine.rs` 串联流
1. **解耦全图 SLANet**：
   * 废除整图直接调用 `table_predictor` 的旧逻辑；
2. **执行双轨流程**：
   * 第一步：全图并发执行 `PP-OCRv6 Det+Rec` 与 `PicoDet-S Layout`；
   * 第二步：提取 `LayoutDetector` 返回的所有 `table` 区域，若无表格则直接跳过 SLANet（极速直通）；
   * 第三步：若有表格，按 ROI 裁剪局部图并调用 `SLANet` 预测局部单元格，随后将单元格坐标平移映射回整图物理坐标系。

### 第三阶段：精简与重构版面排版模块 (`src/layout/`)
1. **大幅瘦身启发式规则**：
   * 彻底废除 `grid_table.rs` 中为抵御全图强切而编写的 1,500+ 行晦涩启发式代码；
2. **轻量空间投影重叠对齐**：
   * 表格区域：由 SLANet 骨架驱动，直接将该局部 ROI 内部包含的 OCR 文本框投影填入对应 `<td></td>`；
   * 非表格区域：由 `para_merger.rs` 负责标准行高定级与段落折行拼接。

### 第四阶段：增强输出契约与 SensiDoc 双向打通
1. **坐标与 AST 增强**：
   * 在 `PageResult` 中显式扩展 `layout_blocks` 与 `cell_boxes`；
2. **SensiDoc 直接集成**：
   * SensiDoc 的 Rust 后台直接调用 `anyocr::parse()`，一举获得：
     * 高质量标准 GFM Markdown；
     * 原件三明治高亮与脱敏所需的全部单元格 `[x1, y1, x2, y2]` 物理坐标。
