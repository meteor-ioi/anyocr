use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// 输入文档格式枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DocumentFormat {
    /// 基于文件头魔数与扩展名自动判定
    #[default]
    Auto,
    /// 常见图像格式 (PNG, JPEG, BMP, WEBP, TIFF)
    Image,
    /// PDF 文档 (含扫描件与双层 PDF)
    Pdf,
    /// 中国国标版式文档 (OFD)
    Ofd,
}

/// 模型规格与场景档案
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ModelProfile {
    /// 极速轻量 (PP-OCRv6-mobile / INT8)，内存低至 ~15MB，适合边缘与 CLI 快速粗扫
    Fast,
    /// 标准平衡 (PP-OCRv6-medium，默认推荐)，兼顾识别率与推理吞吐
    #[default]
    Standard,
    /// 工业高精 (PP-OCRv6-server)，面向财报密集小字、模糊单据与生僻字
    Accurate,
    /// 自定义外部 ONNX 权重路径
    Custom {
        det_path: PathBuf,
        rec_path: PathBuf,
        table_path: Option<PathBuf>,
        dict_path: Option<PathBuf>,
    },
}

/// 硬件加速提供者
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// 智能探测 (Mac 优先 CoreML，Linux 优先 CUDA，回退 CPU)
    #[default]
    Auto,
    /// 纯 CPU SIMD 推理
    Cpu,
    /// Apple Silicon CoreML 神经计算加速
    CoreML,
    /// Nvidia CUDA GPU 加速 (带 GPU 设备编号)
    Cuda(i32),
}

/// 引擎运行时配置
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// 模型尺寸档案 (Fast / Standard / Accurate / Custom)
    pub profile: ModelProfile,
    /// 硬件加速提供者
    pub provider: ExecutionProvider,
    /// 是否开启 SLANet 表格结构识别 (关闭可节省内存与推理时间)
    pub enable_table: bool,
    /// 识别 (Rec) 分桶批处理的最大 Batch Size
    pub max_batch_size: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            profile: ModelProfile::Standard,
            provider: ExecutionProvider::Auto,
            enable_table: true,
            max_batch_size: 16,
        }
    }
}

/// AST 块节点抽象 (借鉴 Pandoc/MinerU，用于结构化版面还原)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocBlock {
    /// 各级标题 (# ~ ######)
    Heading {
        level: u8,
        text: String,
        bbox: [f32; 4],
    },
    /// 自然段落 (跨行折行合并后的文本块)
    Paragraph {
        lines: Vec<TextBoxItem>,
        bbox: [f32; 4],
    },
    /// 结构化表格 (支持 GFM Markdown 管道符与复杂跨行 HTML)
    Table {
        markdown_table: String,
        raw_html: Option<String>,
        bbox: [f32; 4],
    },
    /// 列表项 (包含单项文本与定位坐标)
    List {
        ordered: bool,
        items: Vec<ListItem>,
        bbox: [f32; 4],
    },
    /// 图像/插图/签名/印章区域 (用于多模态与脱敏审计定位)
    Image {
        format: String,
        alt: Option<String>,
        bbox: [f32; 4],
    },
}

/// 列表条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListItem {
    pub text: String,
    pub bbox: [f32; 4],
}

/// 单页解析产物 (支持多页文档如 PDF/OFD 精确定位与流式处理)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageResult {
    /// 当前页码 (0-indexed)
    pub page_index: usize,
    /// 图像/页面主视窗几何尺寸 [宽, 高]
    pub dimensions: (u32, u32),
    /// 本页语义 AST 块集合
    pub blocks: Vec<DocBlock>,
    /// 本页所有检测识别到的单行文本框与置信度 (用于前端原图高亮定位)
    pub boxes: Vec<TextBoxItem>,
}

/// 全文档解析产物
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    /// 结构化排版后的全篇标准 Markdown 文本 (多页以分页符 `\n\n---\n\n` 分隔)
    pub markdown: String,
    /// 多页解析结果清单
    pub pages: Vec<PageResult>,
    /// 总页数
    pub total_pages: usize,
    /// 端到端纯推理与版面还原耗时 (毫秒)
    pub elapsed_ms: u64,
}

/// 基础文本框与置信度元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBoxItem {
    pub text: String,
    pub score: f32,
    /// 空间包围盒坐标: [x1, y1, x2, y2]
    pub coords: [f32; 4],
}
