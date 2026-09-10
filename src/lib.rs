//! # anyocr
//!
//! A native, Python-free Rust OCR & document layout engine powered by ONNX Runtime.
//!
//! 将图片、扫描件 PDF 与国标 OFD 快速解析为标准 Markdown。

pub mod asset;
pub mod engine;
pub mod error;
pub mod ingestion;
pub mod layout;
pub mod models;
pub mod types;

pub use engine::Engine;
pub use error::AnyOcrError;
pub use types::{
    DocBlock, DocumentFormat, EngineConfig, ExecutionProvider, ListItem, ModelProfile,
    PageResult, ParsedDocument, TextBoxItem,
};

static DEFAULT_ENGINE: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();

fn get_default_engine() -> Result<&'static Engine, AnyOcrError> {
    if let Some(engine) = DEFAULT_ENGINE.get() {
        return Ok(engine);
    }
    let engine = Engine::new(EngineConfig::default())?;
    let _ = DEFAULT_ENGINE.set(engine);
    Ok(DEFAULT_ENGINE.get().expect("全局引擎已就绪"))
}

/// 将字节流快速解析为标准 Markdown 字符串 (类似 anydoc 的极简接口)
pub fn to_markdown(bytes: &[u8], format: DocumentFormat) -> Result<String, AnyOcrError> {
    let engine = get_default_engine()?;
    engine.to_markdown(bytes, format)
}

/// 解析字节流并输出包含排版 Markdown 与坐标框的结构化数据 (默认 Standard 配置)
pub fn parse(bytes: &[u8], format: DocumentFormat) -> Result<ParsedDocument, AnyOcrError> {
    let engine = get_default_engine()?;
    engine.parse(bytes, format)
}

/// 基于自定义配置解析字节流 (支持指定模型规格、硬件加速及表格开关)
pub fn parse_with_config(
    bytes: &[u8],
    format: DocumentFormat,
    config: &EngineConfig,
) -> Result<ParsedDocument, AnyOcrError> {
    let engine = Engine::new(config.clone())?;
    engine.parse(bytes, format)
}

