use crate::asset::ModelPaths;
use crate::error::AnyOcrError;
use crate::ingestion::image::ImagePreprocessor;
use crate::models::{LayoutDetector, TextDetector, TextRecognizer};
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
    layout_detector: Option<Arc<LayoutDetector>>,
    #[cfg(feature = "table")]
    #[allow(dead_code)]
    table_predictor: Option<Arc<TableStructurePredictor>>,
}

impl Engine {
    /// 基于指定配置初始化引擎并加载 ONNX 模型
    pub fn new(config: EngineConfig) -> Result<Self, AnyOcrError> {
        let paths = ModelPaths::resolve(&config.profile)?;

        tracing::info!("正在加载 OCR 文本检测模型: {}", paths.det_path.display());
        let detector = Arc::new(TextDetector::from_file(&paths.det_path, config.provider)?);

        tracing::info!("正在加载 OCR 文本识别模型: {}", paths.rec_path.display());
        let recognizer = Arc::new(TextRecognizer::from_files(
            &paths.rec_path,
            &paths.dict_path,
            config.provider,
            config.max_batch_size,
        )?);

        let layout_detector = if let Some(ref l_path) = paths.layout_path {
            tracing::info!("正在加载版面区域分析模型: {}", l_path.display());
            match LayoutDetector::from_file(l_path, config.provider) {
                Ok(ld) => Some(Arc::new(ld)),
                Err(e) => {
                    tracing::warn!("加载版面分析模型失败，降级为规则排版: {e}");
                    None
                }
            }
        } else {
            None
        };

        #[cfg(feature = "table")]
        let table_predictor = if config.enable_table {
            if let Some(ref t_path) = paths.table_path {
                tracing::info!("正在加载表格结构预测模型: {}", t_path.display());
                Some(Arc::new(TableStructurePredictor::from_file(t_path, config.provider)?))
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
            layout_detector,
            #[cfg(feature = "table")]
            table_predictor,
        })
    }

    /// 对 DynamicImage 执行完整的文本检测、文字识别、表格预测与 AST 版面重构 (支持超大图自适应钳制与原图坐标还原)
    pub fn parse_image(&self, img: &DynamicImage) -> Result<ParsedDocument, AnyOcrError> {
        let t0 = Instant::now();
        let (orig_w, orig_h) = (img.width(), img.height());

        // 0. 自适应最长边尺寸保护 (Smart Clamping: 限制超大图内存与计算浪费)
        let (working_img, scale_factor) = if let Some(max_dim) = self.config.max_dimension {
            let max_side = orig_w.max(orig_h);
            if max_side > max_dim && max_side > 0 {
                let factor = max_dim as f32 / max_side as f32;
                let target_w = ((orig_w as f32 * factor).round() as u32).max(1);
                let target_h = ((orig_h as f32 * factor).round() as u32).max(1);
                let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
                (std::borrow::Cow::Owned(resized), factor)
            } else {
                (std::borrow::Cow::Borrowed(img), 1.0f32)
            }
        } else {
            (std::borrow::Cow::Borrowed(img), 1.0f32)
        };

        let (work_w, work_h) = (working_img.width(), working_img.height());

        // 1. 文本行定位检测 (PP-OCRv6 DBNet)
        let t_det = Instant::now();
        let det_boxes = self.detector.detect(&working_img)?;
        let det_ms = t_det.elapsed().as_millis();
        let det_boxes_count = det_boxes.len();

        // 2. 原生版面语义探测 (PicoDet-S Layout)
        let t_layout_det = Instant::now();
        let layout_boxes = if let Some(ref ld) = self.layout_detector {
            ld.detect(&working_img).unwrap_or_default()
        } else {
            Vec::new()
        };
        let _layout_det_ms = t_layout_det.elapsed().as_millis();

        // 3. 提取所有切片 (文本切片 + 表格切片)，提取完成后立即显式释放 working_img 大图
        let mut crops = Vec::with_capacity(det_boxes_count);
        for b in &det_boxes {
            let coords = b.to_array();
            let crop = ImagePreprocessor::crop_box(&working_img, &coords);
            crops.push((crop, coords));
        }

        #[cfg(feature = "table")]
        let mut table_crops = Vec::new();
        #[cfg(feature = "table")]
        if self.config.enable_table && self.table_predictor.is_some() {
            for lb in &layout_boxes {
                if lb.is_table() {
                    let roi = [lb.x1, lb.y1, lb.x2, lb.y2];
                    let crop = ImagePreprocessor::crop_box(&working_img, &roi);
                    table_crops.push((lb.clone(), crop));
                }
            }
        }

        // 💡 显式提前释放超大位图工作缓冲与检测框 (彻底阻断大图与后续批处理识别/表格推理的内存叠加)
        drop(working_img);
        drop(det_boxes);

        // 4. 文本行切片批处理字符识别 (PP-OCRv6 SVTR，带 Arena 内存复用与流式逐批 Drop)
        let t_rec = Instant::now();
        let mut boxes = self.recognizer.recognize_batch(crops);
        let rec_ms = t_rec.elapsed().as_millis();

        // 5. 表格局部 ROI 预测 (SLANet)
        #[cfg(feature = "table")]
        let t_table = Instant::now();
        #[cfg(feature = "table")]
        let table_res_list = if !table_crops.is_empty() {
            let predictor = self.table_predictor.as_ref().unwrap();
            let mut list = Vec::with_capacity(table_crops.len());
            for (lb, crop) in table_crops {
                if let Ok(mut t_res) = predictor.predict(&crop) {
                    for cell in &mut t_res.cell_boxes {
                        cell.x1 += lb.x1;
                        cell.y1 += lb.y1;
                        cell.x2 += lb.x1;
                        cell.y2 += lb.y1;
                    }
                    list.push((lb, t_res));
                }
                drop(crop); // 显式立即释放表格切片
            }
            list
        } else {
            Vec::new()
        };
        #[cfg(feature = "table")]
        let table_ms = t_table.elapsed().as_millis();

        // 6. 端到端 AST 语义语法树与双轨版面还原 (阅读顺序重排、标题定级、表格反填、段落合并)
        let t_layout = Instant::now();
        let (mut blocks, markdown) = crate::layout::LayoutEngine::process_dual_track(
            &boxes,
            (work_w, work_h),
            &layout_boxes,
            #[cfg(feature = "table")]
            &table_res_list,
        );
        let layout_ms = t_layout.elapsed().as_millis();

        // 7. 空间几何坐标逆变换：将 working_img 坐标还原回原图物理像素 (orig_w, orig_h)
        if (scale_factor - 1.0).abs() > 1e-4 && scale_factor > 0.0 {
            for b in &mut boxes {
                b.coords[0] /= scale_factor;
                b.coords[1] /= scale_factor;
                b.coords[2] /= scale_factor;
                b.coords[3] /= scale_factor;
            }
            for block in &mut blocks {
                block.rescale(scale_factor);
            }
        }

        let page = PageResult {
            page_index: 0,
            dimensions: (orig_w, orig_h),
            blocks,
            boxes,
        };

        let elapsed_ms = t0.elapsed().as_millis() as u64;

        #[cfg(feature = "table")]
        eprintln!(
            "[anyocr 阶段耗时] 原图: {}x{} (推理视窗: {}x{}) | 检测: {}ms ({}行) | 识别: {}ms | 表格: {}ms | 排版: {}ms | 端到端: {}ms",
            orig_w, orig_h, work_w, work_h, det_ms, det_boxes_count, rec_ms, table_ms, layout_ms, elapsed_ms
        );
        #[cfg(not(feature = "table"))]
        eprintln!(
            "[anyocr 阶段耗时] 原图: {}x{} (推理视窗: {}x{}) | 检测: {}ms ({}行) | 识别: {}ms | 排版: {}ms | 端到端: {}ms",
            orig_w, orig_h, work_w, work_h, det_ms, det_boxes_count, rec_ms, layout_ms, elapsed_ms
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
