# anyocr

> **A native, Python-free Rust OCR & document layout engine powered by ONNX Runtime.**  
> *(基于 ONNX Runtime 构建的原生、零 Python 依赖 Rust OCR 与版面分析引擎，将图片、扫描件 PDF 与国标 OFD 快速解析为标准 Markdown)*

[![Crates.io](https://img.shields.io/crates/v/anyocr.svg)](https://crates.io/crates/anyocr)
[![Documentation](https://docs.rs/anyocr/badge.svg)](https://docs.rs/anyocr)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

---

## 1. 项目愿景与背景

在当前的 AI Agent、RAG 知识库与本地隐私工具生态中，文档的高保真解析至关重要。
- **现状痛点**：目前优秀的文档版面还原工具（如 MinerU、Docling、Marker、PaddleOCR）几乎全部深度绑定 **Python/PyTorch 运行时**，环境臃肿（动辄数 GB）、部署复杂、冷启动慢，极难嵌入轻量 CLI 工具或桌面客户端；
- **Rust 生态缺失**：Rust 虽有轻量格式解析库（如 `anydoc`），但主要面向 Office 和纯文本，**对纯图片、纸质扫描件、复杂表格与中国国标版式（OFD）无能为力**；
- **核心定位**：`anyocr` 旨在成为 Rust 生态中**开箱即用、零 Python 依赖、直接输出高保真 Markdown 的通用版面感知引擎**。

---

## 2. 核心格式支持矩阵

```text
[ 任意文档输入 ]
       │
       ├── 纯图像类：JPG / JPEG / PNG / BMP / WEBP / TIFF
       ├── 扫描件类：PDF (可编辑文本流直通 / 乱码 CMap 坏死检测 / 内嵌图像 OCR 回退)
       └── 国标版式：OFD (中国电子发票 / 党政公文 / 电子证照)
       │
       ▼
[ anyocr 统一推理与版面重构引擎 ]
       │
       ▼
[ 结构化标准 Markdown (含 GFM 管道符表格 / 标题自适应定级 / 自然段落合并) ]
```

---

## 3. 架构全景

```text
                     [ anyocr 双层 API 架构 ]
          ├── 极简脚本：anyocr::to_markdown(bytes, format) -> String
          └── 高吞吐服务：engine = Engine::new(config)?; engine.parse(...)
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
 [ Image Pipeline ]      [ PDF Ingestion ]       [ OFD Ingestion ]
 (动态缩放/格式解码)     (文本流直通/坏死层熔断)   (XML提取/内嵌图分流)
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               ▼
        [ ONNX Runtime Inference Engine (Native Rust) ]
         ├── Text Detection    : PP-OCRv6-small DBNet (RepLKFPN)
         ├── Text Recognition  : PP-OCRv6 SVTR + 长宽比分桶批处理 (Bucket Batching)
         └── Table Recognition : SLANet_plus 50 词元序列预测 (可选 Feature)
                               ▼
        [ Layout & AST Post-processor (启发式版面还原引擎) ]
         ├── 段落折行重构 (LINE_STOP_FLAGS + 中文去空格/英文单词智能粘合)
         ├── 自适应行高定级 (H_body 中位数自适应聚类 ➔ 映射 # / ## / ###)
         ├── 表格空间反填 (SLANet 单元格与文字拓扑对齐 ➔ GFM 管道符表格)
         └── 多栏阅读排序 (X 轴投影直方图检测双栏中缝，根治左右串行)
                               │
                               ▼
            [ Clean GFM Markdown Output + PageResult AST ]
```

---

## 4. 快速上手

在 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
anyocr = "0.1"
```

### 方式一：极简开箱即用（类似 anydoc，适合脚本与 CLI）

```rust
use anyocr::{to_markdown, DocumentFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("invoice.png")?;
    
    // 传入 DocumentFormat::Auto 会自动基于文件魔数探测格式 (PDF / OFD / Image)
    let markdown = to_markdown(&bytes, DocumentFormat::Auto)?;
    println!("{}", markdown);
    Ok(())
}
```

### 方式二：工业级显式持有（推荐：高吞吐服务与多任务并发复用）

```rust
use anyocr::{Engine, EngineConfig, ModelProfile, ExecutionProvider, DocumentFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 一次性初始化引擎并加载 ONNX 运行时 Session，避免重复冷启动
    let config = EngineConfig {
        profile: ModelProfile::Standard,       // Fast / Standard / Accurate / Custom
        provider: ExecutionProvider::Auto,     // Mac CoreML / Nvidia CUDA / CPU SIMD
        enable_table: true,                    // 开启 SLANet 表格结构识别
        max_batch_size: 16,                    // 分桶批处理并发度
    };
    let engine = Engine::new(config)?;

    // 2. 多次调用，复用温暖的会话实例
    let bytes = std::fs::read("financial_report.pdf")?;
    let doc = engine.parse(&bytes, DocumentFormat::Pdf)?;

    println!("总页数: {}, 解析耗时: {}ms", doc.total_pages, doc.elapsed_ms);
    for page in &doc.pages {
        println!("第 {} 页检出 {} 个文本框, {} 个 AST 语法块", page.page_index + 1, page.boxes.len(), page.blocks.len());
    }

    println!("\nMarkdown:\n{}", doc.markdown);
    Ok(())
}
```

---

## 5. Feature Flags 裁剪说明

`anyocr` 提供了细粒度的 Cargo Feature Flags，方便根据部署环境极度瘦身：

| Feature | 默认开启 | 说明 |
| :--- | :---: | :--- |
| `pdf` | ✅ | 基于 `lopdf` 的 PDF 文本流直通与扫描件提取 |
| `ofd` | ✅ | 中国国标版式 OFD 容器解包与 XML 文本提取 |
| `table` | ✅ | 基于 `SLANet_plus` 的表格拓扑与单元格反填（关闭可节省内存） |
| `download` | ✅ | 模型按需下载器与本地缓存校验支持 |

如果你只需要纯图片 OCR 且处于完全封闭的离线内网：
```toml
[dependencies]
anyocr = { version = "0.1", default-features = false }
```

---

## 6. 模型规格与场景档案 (Model Profiles)

| Profile | 规格推荐 | 内存占用 | 适用场景 |
| :--- | :--- | :---: | :--- |
| `ModelProfile::Fast` | PP-OCRv6-mobile / INT8 | ~15MB | 边缘设备、轻量 CLI 粗扫、低配云主机 |
| `ModelProfile::Standard` | PP-OCRv6-medium (默认) | ~80MB | 通用文档、增值税发票、合同单据、公文 |
| `ModelProfile::Accurate` | PP-OCRv6-server | ~160MB | 密集数字财报、模糊翻拍单据、专业生僻字 |
| `ModelProfile::Custom` | 外部自定义权重 | 自定义 | 挂载企业微调私有模型与特定语种字典 |

---

## 7. 许可证

本项目遵循 [MIT License](LICENSE-MIT) 或 [Apache-2.0 License](LICENSE-APACHE)。
