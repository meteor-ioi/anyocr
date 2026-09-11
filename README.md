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
| `ModelProfile::Fast` | PP-OCRv6-small (默认推荐) | ~20MB | 极速响应 (~500ms)、高并发服务、轻量 CLI 交互 |
| `ModelProfile::Standard` | PP-OCRv6-medium | ~80MB | 通用文档、古籍生僻字、超模糊极端抗噪场景 |
| `ModelProfile::Accurate` | PP-OCRv6-server | ~160MB | 密集数字财报、超高精度翻拍单据 |
| `ModelProfile::Custom` | 外部自定义权重 | 自定义 | 挂载企业私有微调模型与小语种专有识别器 |

---

## 7. 多语种与小语种扩展指南 (Multilingual Support)

`anyocr` 的文本定位检测（DBNet）与表格拓扑预测（SLANet）均为**语言无关（Language-Agnostic）**设计，无论输入是何种语言，均能精准定位文字行并还原版面。

### 7.1 原生内置语种（零额外依赖）
默认预置的 `PP-OCRv6` 字典已混合覆盖 **18,710 字符**：
- **中文**：简体中文通用全量字库、繁体中文、生僻字与异体字；
- **英文及拉丁语族**：英语、法语、德语、意大利语、西班牙语、葡萄牙语等完整变音重音符；
- **日文**：日文汉字、平假名、片假名。

### 7.2 扩展其他 80+ 种全球小语种
对于韩语、俄语、泰语、阿拉伯语、印地语等特定小语种，可直接复用 [RapidAI 魔搭社区 (ModelScope)](https://www.modelscope.cn/models/RapidAI/RapidOCR/files) 预先转换好的开箱即用 `.onnx` 模型与配套字典。

#### 步骤一：从魔搭下载小语种 ONNX 权重与字典
以韩语（Korean）为例：
```bash
# 1. 下载韩语 ONNX 识别模型 (约 10MB)
curl -L "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx" \
     -o models/ocr/korean_PP-OCRv5_rec_mobile.onnx

# 2. 下载配对字典
curl -L "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/paddle/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile/ppocrv5_korean_dict.txt" \
     -o models/ocr/ppocrv5_korean_dict.txt
```

> 常用语种目录对照（魔搭 `RapidAI/RapidOCR` 仓库）：
> - **韩语**：`onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx`
> - **泰语**：`onnx/PP-OCRv5/rec/th_PP-OCRv5_rec_mobile.onnx`
> - **拉丁扩展 (越/波/捷/土等)**：`onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx`
> - **泰米尔语 / 泰卢固语**：`onnx/PP-OCRv5/rec/ta_...` / `te_...`

#### 步骤二：通过 `ModelProfile::Custom` 零代码无缝挂载
```rust
use anyocr::{Engine, EngineConfig, ModelProfile, DocumentFormat};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig {
        profile: ModelProfile::Custom {
            // 复用通用超轻量检测与表格模型
            det_path: PathBuf::from("models/ocr/PP-OCRv6_det_small.onnx"),
            table_path: Some(PathBuf::from("models/ocr/slanet-plus.onnx")),
            // 挂载小语种识别器与字典
            rec_path: PathBuf::from("models/ocr/korean_PP-OCRv5_rec_mobile.onnx"),
            dict_path: Some(PathBuf::from("models/ocr/ppocrv5_korean_dict.txt")),
        },
        ..Default::default()
    };

    let engine = Engine::new(config)?;
    let doc = engine.parse(&std::fs::read("korean_doc.pdf")?, DocumentFormat::Pdf)?;
    println!("{}", doc.markdown);
    Ok(())
}
```

---

## 8. 许可证

本项目遵循 [MIT License](LICENSE-MIT) 或 [Apache-2.0 License](LICENSE-APACHE)。

