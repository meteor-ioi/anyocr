# anyocr 核心开发方案与架构指南

> **目标受众**：本规范专为接手 `anyocr` 项目的开发者与 AI Agent 编写。请在编码前通读本指南，严格遵循本设计的技术边界、算法原则与分层架构。

---

## 一、 项目背景与使命定位

### 1.1 项目起源
本模块最初孵化于本地敏感信息智能审计工具 [`SensiDoc`](https://github.com/meteor-ioi/SensiDoc) 的底层 OCR 模组。在实际演进中，团队发现“将图片、扫描件 PDF、OFD 等版式文档高质量还原为 Markdown”是一个极其通用的核心基础能力，而不仅仅服务于脱敏审计。

### 1.2 填补生态空白
- **与 `anydoc` 形成黄金搭档**：`anydoc` 擅长纯文本与 Office（Word/Excel/PPT），但遇到图片、扫描件直接报错或留白；`anyocr` 则专注解决视觉扫描件、复杂单据与版式文档的结构化解析；
- **Native & Zero-Python**：彻底摆脱 Python、PyTorch、Paddle 等笨重环境，纯 Rust 原生嵌入，提供毫秒级冷启动与极低资源占用；
- **攻克国标 OFD**：原生支持中国电子发票、党政公文、电子证照等 OFD 格式的文本提取与扫描图 OCR 路由。

---

## 二、 前置知识储备（Agent 必读）

### 2.1 依赖生态与关键 Crate
| Crate | 作用与关键点 |
| :--- | :--- |
| `ort` (v2.0.0-rc.9+) | ONNX Runtime 的 Rust 原生绑定。负责加载与执行 ONNX 计算图，支持 CPU SIMD、Metal 与 CUDA。 |
| `image` (v0.25+) | 纯 Rust 图像解码与像素操作库。支持 PNG, JPEG, BMP, WEBP 等。 |
| `ndarray` (v0.17+) | 多维数组计算，用于构造模型 NCHW 输入张量与后处理矩阵操作。 |
| `quick-xml` + `zip` | OFD 格式本质是 ZIP 容器包装的 XML 描述文件，用于高效无损流式解压与解析。 |
| `lopdf` | 轻量 PDF 解析，用于提取可编辑文本层或内嵌扫描图像。 |

### 2.2 核心深度学习模型说明与多尺寸分级 (ModelProfile)
为满足边缘轻量、桌面通用与工业高精的不同场景需求，引擎提供三档预置档案并支持外部自定义权重：
- **`ModelProfile::Fast`（轻量极速）**：
  - Rec 选用 `PP-OCRv6-mobile` / INT8 量化版，Det 选用轻量 DBNet；
  - 内存占用仅 ~15MB，适合 CLI、低配置容器、秒级快速粗扫。
- **`ModelProfile::Standard`（默认平衡，开箱即用）**：
  - Rec 选用 `PP-OCRv6-medium`，Det 选用标准 DBNet；
  - 平衡速度与 98%+ 常见文档的高准确率。
- **`ModelProfile::Accurate`（专业高精）**：
  - Rec 选用 `PP-OCRv6-server` 级大模型；
  - 针对财务报表密集小字、模糊复印件、低对比度单据、繁体与生僻字，提供极高的文字召回率。
- **`ModelProfile::Custom`**：
  - 允许外部传入自定义 ONNX 模型路径及字典（`keys.txt`）。

表格结构预测模型选用 `SLANet_plus`（固定输入 `488x488`），输出 HTML Tokens 序列与单元格相对坐标框。

### 2.3 推理执行优化：长宽比分桶与硬件加速
1. **长宽比分桶批处理 (Bucket Batching)**：
   - DBNet 裁剪出的多行文字框宽高比差异极大，若逐行串行推理则 ONNX 调度开销过大；若统一切齐 Padding 会浪费大量无效算力。
   - 将文本行按宽高比动态划入 3~4 个桶（Short / Medium / Long），以 Batch Size（如 8 或 16）打包推理，吞吐量提升 2~3 倍。
2. **硬件执行提供者 (Execution Providers)**：
   - 抽象 `ExecutionProvider`：支持 `Cpu`、`Auto`（Mac 优先 `CoreML`，Linux 优先 `Cuda`）、`Cuda(device_id)`、`CoreML`。

### 2.4 版面重构与 AST 中间表示（借鉴 MinerU / Docling）
模型解决“字是什么、在哪里”，版面引擎负责将其转化为具备层次的文档结构：
1. **AST 中间表示 (DocBlock)**：
   - 拒绝简单的纯文本字符串拼凑，先构造抽象语法树节点：`Heading(level, text)`、`Paragraph(lines)`、`Table(grid)`、`List(items)`，再统一序列化为 Markdown 或保留前端高亮坐标。
2. **段落合并与跨行折行 (Para Split & Merge)**：
   - 截断标点集：`LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ':', '：', ';', '；')`；
   - 若上一行末尾不含断句标点且下一行左侧对齐，判定为自然段跨行折叠，消除满屏滥用 `- 列表项`；
   - 英文折行智能补空格与处理连字符，中文折行剔除多余空格。
3. **自适应行高聚类 (Heading Leveling)**：
   - 统计单页文本框高度中位数作为基准正文字号 $H_{body}$；
   - $H > 1.8 \times H_{body}$ 映射为一级标题 `# `；
   - $1.4 \times H_{body} < H \le 1.8 \times H_{body}$ 映射为二级标题 `## `。
4. **多栏阅读顺序重构 (Reading Order)**：
   - 采用 X 轴投影直方图寻找垂直中缝空白谷底（White-space Valley）；
   - 双栏拓扑排序：`顶通栏 ➔ 左栏 ➔ 右栏 ➔ 底通栏`，根治横向串行。
5. **表格空间反填 (Table Spatial Infilling)**：
   - 计算 OCR 文本行中心点/面积与 SLANet 预测的单元格 BBox 的空间交并比（IoU）；
   - 规则表格输出标准 Markdown 管道符（`| 列1 | 列2 |`），复杂跨行跨列内嵌安全 `<table>` 标签。
6. **PDF 坏死文本层自动熔断与回退**：
   - 针对 ToUnicode CMap 丢失或损坏的 PDF，统计乱码率与空白字符比率，一旦异常自动回退为“渲染/提取内嵌图 ➔ OCR 识别”。

---

## 三、 架构设计与目录划分

```text
anyocr/
├── Cargo.toml                  (库定义与细粒度 features: [pdf, ofd, table, download, coreml, cuda])
├── README.md                   (全局说明与快速入门)
├── DEVELOPMENT_PLAN.md         (开发方案与架构指南)
├── todo.md                     (实施计划任务清单)
├── src/
│   ├── lib.rs                  (公开入口，导出 to_markdown, parse, parse_with_config)
│   ├── error.rs                (统一强类型错误定义)
│   ├── types.rs                (ParsedDocument, TextBoxItem, ModelProfile, EngineConfig, DocBlock)
│   ├── engine.rs               (Engine 运行时生命周期、分桶批处理与模型单例管理)
│   ├── ingestion/              (文档摄入与预处理)
│   │   ├── mod.rs
│   │   ├── image.rs            (图像解码、格式校验与长宽缩放)
│   │   ├── pdf.rs              (PDF 文本层提取、坏死字符检测与扫描件图片流抽取)
│   │   └── ofd.rs              (OFD 解压、XML <ofd:TextCode> 抽取与扫描图分流)
│   ├── models/                 (ONNX 模型前向推理)
│   │   ├── mod.rs
│   │   ├── detector.rs         (PP-OCRv6 DBNet 文本定位)
│   │   ├── recognizer.rs       (PP-OCRv6 SVTR/CRNN 多规格字符识别与 CTC 解码，支持分桶批处理)
│   │   └── table.rs            (SLANet_plus 表格结构预测)
│   ├── layout/                 (版面还原与语法树后处理)
│   │   ├── mod.rs
│   │   ├── ast.rs              (DocBlock 抽象语法树表示)
│   │   ├── para_merger.rs      (标点截断、缩进与段落合并)
│   │   ├── heading.rs          (行高聚类中位数统计与标题定级)
│   │   ├── reading_order.rs    (X 轴直方图投影与分栏拓扑排序)
│   │   ├── table_matcher.rs    (空间几何拓扑与单元格反填)
│   │   └── md_serializer.rs    (AST 语法树序列化为干净 Markdown)
│   └── asset/                  (模型资源管理)
│       ├── mod.rs
│       └── downloader.rs       (轻量内置模型自动下载与本地缓存机制)
└── examples/
    └── simple_convert.rs       (10 行代码快速使用演示)
```

---

## 四、 公共 API 规范设计

作为基础设施库，接口采用**双层 API 设计**，兼顾“开箱即用极简体验”与“企业级服务高并发复用”：

```rust
// -------------------------------------------------------------
// 1. 核心工业级接口 (推荐：显式管理引擎生命周期与模型内存复用)
// -------------------------------------------------------------
let engine = anyocr::Engine::new(EngineConfig::default())?;
let doc = engine.parse(bytes, DocumentFormat::Auto)?;
let md = engine.to_markdown(bytes, DocumentFormat::Image)?;

// -------------------------------------------------------------
// 2. 便捷静态接口 (内部维护全局单例，适合快速脚本与 CLI 工具)
// -------------------------------------------------------------
pub fn to_markdown(bytes: &[u8], format: DocumentFormat) -> Result<String, AnyOcrError>;
pub fn parse(bytes: &[u8], format: DocumentFormat) -> Result<ParsedDocument, AnyOcrError>;
pub fn parse_with_config(
    bytes: &[u8],
    format: DocumentFormat,
    config: &EngineConfig,
) -> Result<ParsedDocument, AnyOcrError>;

/// 引擎运行时配置
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub profile: ModelProfile,
    pub provider: ExecutionProvider,
    pub enable_table: bool,
    pub max_batch_size: usize,
}

/// 模型规格与场景档案
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ModelProfile {
    /// 极速轻量 (PP-OCRv6-mobile / INT8)，内存低至 ~15MB
    Fast,
    /// 标准平衡 (PP-OCRv6-medium，默认推荐)，高准确率
    #[default]
    Standard,
    /// 工业高精 (PP-OCRv6-server)，面向财报密集小字、模糊单据
    Accurate,
    /// 自定义外部权重路径
    Custom {
        det_path: std::path::PathBuf,
        rec_path: std::path::PathBuf,
        table_path: Option<std::path::PathBuf>,
        dict_path: Option<std::path::PathBuf>,
    },
}

/// 硬件加速提供者
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    #[default]
    Auto,       // Mac 优先 CoreML，Linux 优先 CUDA，回退 CPU
    Cpu,
    CoreML,
    Cuda(i32),
}

/// 输入格式枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DocumentFormat {
    #[default]
    Auto,           // 基于文件头部魔数自动探测
    Image,          // 纯图像 (JPG, PNG, WEBP, TIFF 等)
    Pdf,            // PDF 文档 (支持多页与双层 PDF)
    Ofd,            // 中国国标版式 OFD
}

/// AST 块节点抽象
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocBlock {
    Heading { level: u8, text: String, bbox: [f32; 4] },
    Paragraph { lines: Vec<TextBoxItem>, bbox: [f32; 4] },
    Table { markdown_table: String, raw_html: Option<String>, bbox: [f32; 4] },
    List { ordered: bool, items: Vec<ListItem>, bbox: [f32; 4] },
    Image { format: String, alt: Option<String>, bbox: [f32; 4] },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListItem {
    pub text: String,
    pub bbox: [f32; 4],
}

/// 单页解析产物 (支持多页 PDF/OFD 独立空间坐标与流式处理)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageResult {
    pub page_index: usize,
    pub dimensions: (u32, u32),
    pub blocks: Vec<DocBlock>,
    pub boxes: Vec<TextBoxItem>,
}

/// 全文档解析产物
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    pub markdown: String,
    pub pages: Vec<PageResult>,
    pub total_pages: usize,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBoxItem {
    pub text: String,
    pub score: f32,
    pub coords: [f32; 4], // [x1, y1, x2, y2]
}
```

---

## 五、 分阶段实施路线图（Milestones）

### Milestone 1：初始化骨架、图像摄入与 OCR 资产迁移（跑通单图闭环）
- [x] 细化 `Cargo.toml`（配置 `pdf`, `ofd`, `table`, `download` 及 `image` 的 `tiff` 支持）；
- [x] 完善 `types.rs` 与 `error.rs`（支持 `Engine`, `PageResult`, `ModelProfile`, `DocBlock`, `UnsupportedFormat`）；
- [ ] 实现 `ingestion/image.rs`：纯图像格式解码（JPG/PNG/BMP/WEBP/TIFF）、EXIF 旋转矫正与缩放；
- [ ] 借鉴并重构 SensiDoc 原有的 `src/ocr/` 成熟代码资产：
  - 迁移 `text_detector.rs` ➔ `models/detector.rs`；
  - 迁移 `text_recognizer.rs` ➔ `models/recognizer.rs`（先建立逐行 baseline 推理）；
  - 迁移 `table_structure.rs` ➔ `models/table.rs`；
- [ ] 实现 `asset/downloader.rs`：支持根据 `ModelProfile` 按需自动下载模型权重并校验 SHA256；
- [ ] 实现 `engine.rs` 基础生命周期管理，编写单张图片的端到端冒烟测试。

### Milestone 2：版面还原、AST 语法树重构与分桶批处理优化（质的跃升）
- [ ] 实现 `layout/ast.rs`：定义与组装 `DocBlock` 文档语法树；
- [ ] 实现 `layout/para_merger.rs`：基于 `LINE_STOP_FLAG` 消除满屏无序列表 `- ` 碎片；
- [ ] 实现 `layout/heading.rs`：自适应行高聚类，统计中位数输出 `# / ## / ###`；
- [ ] 整合 `layout/table_matcher.rs`：将单元格与文字空间对齐，并优先输出 GFM 管道符表格；
- [ ] 实现 `layout/md_serializer.rs`：将 `DocBlock` 树渲染为纯净 Markdown；
- [ ] **吞吐优化**：在 `models/recognizer.rs` 中落地长宽比分桶批处理（Bucket Batching），提升推理吞吐 2~3 倍；
- [ ] 编写端到端单测验证复杂发票与制度单据的 Markdown 输出质量。

### Milestone 3：多页 PDF 扫描件与国标 OFD 集成（全格式覆盖）
- [ ] 实现 `ingestion/pdf.rs`：
  - 基于 `lopdf` 快速抽取可编辑文字流并构建 `PageResult`；
  - 增加坏死乱码检测（统计未映射字体与不可见字符比例）；
  - 扫描件 PDF 提取内嵌图像流走 OCR，输出页级坐标；
- [ ] 实现 `ingestion/ofd.rs`：
  - 解析 OFD ZIP 包结构，流式提取 `<ofd:TextCode>` 文本与坐标；
  - 若是图片型发票/公文，抽取内嵌图片走 OCR；
- [ ] 完成统一入口 `to_markdown`, `parse`, `parse_with_config` 的全格式自动路由。

### Milestone 4：SensiDoc 回归集成与开源发布
- [ ] 在 `SensiDoc` 中通过本地路径引入 `anyocr`，彻底删除 SensiDoc 内部冗余的 `src/ocr/` 代码；
- [ ] 验证 SensiDoc 的 OCR 识别、前端高亮与脱敏链路完全保持兼容；
- [ ] 完善开源仓库 README，补充双语文档与 GitHub Actions CI 流水线，准备发布至 crates.io。

---

## 六、 既有资产参考指南

在 `SensiDoc` 原工程中已有高度可用的现成代码，可直接参考与重构：
- `/Users/icychick/Projects/SensiDoc-ocr/src/ocr/text_detector.rs` (已验证的 DBNet 图像前处理与二值化多边形提取)
- `/Users/icychick/Projects/SensiDoc-ocr/src/ocr/text_recognizer.rs` (已验证的 SVTR 动态批推理与 CTC 字典映射)
- `/Users/icychick/Projects/SensiDoc-ocr/src/ocr/table_structure.rs` (已验证的 SLANet_plus 结构预测)
- `/Users/icychick/Projects/SensiDoc-ocr/src/ocr/matcher.rs` (已具备行级单调拓扑对齐雏形)
- `/Users/icychick/Projects/SensiDoc-ocr/src/ocr/markdown_builder.rs` (GFM 表格转换逻辑)
- 调研方案参考：`/Users/icychick/Projects/SensiDoc-ocr/plan/OCR_TO_MARKDOWN_STRUCTURE_PARSING_RESEARCH.md`
