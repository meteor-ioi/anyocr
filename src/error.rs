use thiserror::Error;

#[derive(Error, Debug)]
pub enum AnyOcrError {
    #[error("图像解码或处理错误: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("ONNX 推理引擎错误: {0}")]
    InferenceError(String),

    #[error("不支持或未启用该格式的解析特性: {0} (请检查 Cargo.toml 是否开启对应 feature)")]
    UnsupportedFormat(String),

    #[error("PDF 解析错误: {0}")]
    PdfError(String),

    #[error("OFD 国标版式解析错误: {0}")]
    OfdError(String),

    #[error("模型资产未就绪: {0}")]
    ModelNotReady(String),

    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("其他未知错误: {0}")]
    Other(String),
}
