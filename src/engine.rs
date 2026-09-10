use crate::asset::ModelPaths;
use crate::error::AnyOcrError;
use crate::ingestion::image::ImagePreprocessor;
use crate::models::{TextDetector, TextRecognizer};
#[cfg(feature = "table")]
use crate::models::TableStructurePredictor;
use crate::types::{
    DocumentFormat, EngineConfig, PageResult, ParsedDocument,
};
use image::DynamicImage;
use std::sync::Arc;
use std::time::Instant;

/// OCR 核心运行时引擎实例 (显式持有 Session 会话生命周期与硬件连接)
pub struct Engine {
    config: EngineConfig,
    detector: Arc<TextDetector>,
    recognizer: Arc<TextRecognizer>,
    #[cfg(feature = "table")]
    #[allow(dead_code)]
    table_predictor: Option<Arc<TableStructurePredictor>>,
}

impl Engine {
    /// 基于指定配置初始化引擎并加载 ONNX 模型
    pub fn new(config: EngineConfig) -> Result<Self, AnyOcrError> {
        let paths = ModelPaths::resolve(&config.profile)?;

        tracing::info!("正在加载 OCR 文本检测模型: {}", paths.det_path.display());
        let detector = Arc::new(TextDetector::from_file_with_provider(&paths.det_path, config.provider)?);

        tracing::info!("正在加载 OCR 文本识别模型: {}", paths.rec_path.display());
        let recognizer = Arc::new(TextRecognizer::from_files_with_provider(
            &paths.rec_path,
            &paths.dict_path,
            config.provider,
            config.max_batch_size,
        )?);

        #[cfg(feature = "table")]
        let table_predictor = if config.enable_table {
            if let Some(ref t_path) = paths.table_path {
                tracing::info!("正在加载表格结构预测模型: {}", t_path.display());
                Some(Arc::new(TableStructurePredictor::from_file_with_provider(t_path, config.provider)?))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            detector,
            recognizer,
            #[cfg(feature = "table")]
            table_predictor,
        })
    }

    /// 对 DynamicImage 执行完整的文本检测、文字识别、表格预测与 AST 版面重构
    pub fn parse_image(&self, img: &DynamicImage) -> Result<ParsedDocument, AnyOcrError> {
        let t0 = Instant::now();
        let (img_w, img_h) = (img.width(), img.height());

        // 1. 文本行定位检测 (PP-OCRv6 DBNet)
        let t_det = Instant::now();
        let det_boxes = self.detector.detect(img)?;
        let det_ms = t_det.elapsed().as_millis();

        // 2. 文本行切片裁剪与字符识别 (PP-OCRv6 SVTR)
        let t_rec = Instant::now();
        use rayon::prelude::*;
        let crops: Vec<(DynamicImage, [f32; 4])> = det_boxes
            .par_iter()
            .map(|b| {
                let coords = b.to_array();
                let crop = ImagePreprocessor::crop_box(img, &coords);
                (crop, coords)
            })
            .collect();
        let boxes = self.recognizer.recognize_batch(&crops);
        let rec_ms = t_rec.elapsed().as_millis();

        // 3. 表格结构预测 (若开启 table feature 且引擎配置启用)
        #[cfg(feature = "table")]
        let t_table = Instant::now();
        #[cfg(feature = "table")]
        let table_res = if self.config.enable_table {
            self.table_predictor.as_ref().and_then(|p| {
                match p.predict(img) {
                    Ok(res) => Some(res),
                    Err(e) => {
                        tracing::warn!("表格结构预测失败，将降级为常规文本排版: {e}");
                        None
                    }
                }
            })
        } else {
            None
        };
        #[cfg(feature = "table")]
        let table_ms = t_table.elapsed().as_millis();

        // 4. 端到端 AST 语义语法树与版面还原 (阅读顺序重排、标题定级、表格反填、段落合并)
        let t_layout = Instant::now();
        let (blocks, markdown) = crate::layout::LayoutEngine::process(
            &boxes,
            (img_w, img_h),
            #[cfg(feature = "table")]
            table_res.as_ref(),
        );
        let layout_ms = t_layout.elapsed().as_millis();

        let page = PageResult {
            page_index: 0,
            dimensions: (img_w, img_h),
            blocks,
            boxes,
        };

        let elapsed_ms = t0.elapsed().as_millis() as u64;

        #[cfg(feature = "table")]
        eprintln!(
            "[anyocr 阶段耗时] 分辨率: {}x{} | 检测: {}ms ({}行) | 识别: {}ms | 表格: {}ms | 排版: {}ms | 端到端: {}ms",
            img_w, img_h, det_ms, det_boxes.len(), rec_ms, table_ms, layout_ms, elapsed_ms
        );
        #[cfg(not(feature = "table"))]
        eprintln!(
            "[anyocr 阶段耗时] 分辨率: {}x{} | 检测: {}ms ({}行) | 识别: {}ms | 排版: {}ms | 端到端: {}ms",
            img_w, img_h, det_ms, det_boxes.len(), rec_ms, layout_ms, elapsed_ms
        );

        Ok(ParsedDocument {
            markdown,
            pages: vec![page],
            total_pages: 1,
            elapsed_ms,
        })
    }

    /// 基于文件头魔数自动探测输入文档格式
    pub fn detect_format(bytes: &[u8]) -> DocumentFormat {
        if bytes.starts_with(b"%PDF-") {
            return DocumentFormat::Pdf;
        }

        // ZIP 压缩包容器检测 (OFD 为国标 ZIP 打包格式)
        if bytes.starts_with(b"PK\x03\x04") {
            // 扫描前 4096 字节是否包含 OFD.xml 标记
            let probe_len = bytes.len().min(4096);
            let slice = &bytes[..probe_len];
            if slice.windows(7).any(|w| w == b"OFD.xml" || w == b"ofd.xml") {
                return DocumentFormat::Ofd;
            }
        }

        DocumentFormat::Image
    }

    /// 解析输入文档字节流并输出结构化排版结果 (支持全格式自动识别与路由)
    pub fn parse(&self, bytes: &[u8], format: DocumentFormat) -> Result<ParsedDocument, AnyOcrError> {
        let target_format = if format == DocumentFormat::Auto {
            Self::detect_format(bytes)
        } else {
            format
        };

        match target_format {
            DocumentFormat::Image | DocumentFormat::Auto => {
                let img = ImagePreprocessor::decode_image(bytes)?;
                self.parse_image(&img)
            }
            DocumentFormat::Pdf => {
                #[cfg(feature = "pdf")]
                {
                    crate::ingestion::PdfIngestion::parse_pdf(bytes, self)
                }
                #[cfg(not(feature = "pdf"))]
                {
                    Err(AnyOcrError::UnsupportedFormat(
                        "PDF 解析未开启 'pdf' feature，请在 Cargo.toml 中开启对应依赖".to_string(),
                    ))
                }
            }
            DocumentFormat::Ofd => {
                #[cfg(feature = "ofd")]
                {
                    crate::ingestion::OfdIngestion::parse_ofd(bytes, self)
                }
                #[cfg(not(feature = "ofd"))]
                {
                    Err(AnyOcrError::UnsupportedFormat(
                        "OFD 解析未开启 'ofd' feature，请在 Cargo.toml 中开启对应依赖".to_string(),
                    ))
                }
            }
        }
    }

    /// 解析输入文档字节流并直接输出排版好的 Markdown 文本
    pub fn to_markdown(&self, bytes: &[u8], format: DocumentFormat) -> Result<String, AnyOcrError> {
        let doc = self.parse(bytes, format)?;
        Ok(doc.markdown)
    }

    /// 获取当前引擎配置引用
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_engine_smoke_on_real_image() {
        let sample_path = "/Users/icychick/.gemini/antigravity-cli/brain/6efdb7c6-eaca-4a22-8acc-97baeea17ede/.user_uploaded/uploaded_media_1789028772708.png";
        if !Path::new(sample_path).exists() {
            println!("测试样本图不存在，跳过冒烟");
            return;
        }

        let config = EngineConfig::default();
        let engine = match Engine::new(config) {
            Ok(e) => e,
            Err(e) => {
                println!("本地模型未就绪，跳过真实推理冒烟测试: {e}");
                return;
            }
        };

        let bytes = std::fs::read(sample_path).expect("读取测试图片失败");
        let result = engine.parse(&bytes, DocumentFormat::Image).expect("端到端单图推理失败");

        println!("\n=== anyocr 单图端到端冒烟测试输出 ===");
        println!("{}", result.markdown);
        println!("=== 耗时: {}ms, 检出文字框数量: {} ===", result.elapsed_ms, result.pages[0].boxes.len());

        assert!(!result.markdown.is_empty(), "产出 Markdown 不应为空");
        assert!(!result.pages.is_empty(), "单图应产出 1 页结果");
        assert!(!result.pages[0].boxes.is_empty(), "应检出文本框");
    }
}
