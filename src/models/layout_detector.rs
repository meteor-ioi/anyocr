use crate::error::AnyOcrError;
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

/// PicoDet-S 预测出的版面区域标签
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LayoutLabel {
    ParagraphTitle,
    Image,
    Text,
    Number,
    Abstract,
    Content,
    FigureTitle,
    Formula,
    Table,
    TableTitle,
    Reference,
    DocTitle,
    Footnote,
    Header,
    Algorithm,
    Footer,
    Seal,
    Unknown,
}

impl LayoutLabel {
    pub fn from_class_id(id: usize) -> Self {
        match id {
            0 => Self::ParagraphTitle,
            1 => Self::Image,
            2 => Self::Text,
            3 => Self::Number,
            4 => Self::Abstract,
            5 => Self::Content,
            6 => Self::FigureTitle,
            7 => Self::Formula,
            8 => Self::Table,
            9 => Self::TableTitle,
            10 => Self::Reference,
            11 => Self::DocTitle,
            12 => Self::Footnote,
            13 => Self::Header,
            14 => Self::Algorithm,
            15 => Self::Footer,
            16 => Self::Seal,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ParagraphTitle => "paragraph_title",
            Self::Image => "image",
            Self::Text => "text",
            Self::Number => "number",
            Self::Abstract => "abstract",
            Self::Content => "content",
            Self::FigureTitle => "figure_title",
            Self::Formula => "formula",
            Self::Table => "table",
            Self::TableTitle => "table_title",
            Self::Reference => "reference",
            Self::DocTitle => "doc_title",
            Self::Footnote => "footnote",
            Self::Header => "header",
            Self::Algorithm => "algorithm",
            Self::Footer => "footer",
            Self::Seal => "seal",
            Self::Unknown => "unknown",
        }
    }
}

/// 单个检测到的版面区域及其物理外接矩形 [x1, y1, x2, y2]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayoutBox {
    pub label: LayoutLabel,
    pub score: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl LayoutBox {
    pub fn to_array(&self) -> [f32; 4] {
        [self.x1, self.y1, self.x2, self.y2]
    }

    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn is_table(&self) -> bool {
        self.label == LayoutLabel::Table
    }
}

/// 基于 PicoDet-S 的原生极轻量文档版面分析检测器 (4.9MB ONNX)
pub struct LayoutDetector {
    session: Mutex<Session>,
    score_thresh: f32,
    target_size: (u32, u32),
}

impl LayoutDetector {
    /// 从 ONNX 模型文件构建版面检测器
    pub fn from_file(
        model_path: impl AsRef<Path>,
        provider: crate::types::ExecutionProvider,
    ) -> Result<Self, AnyOcrError> {
        let session = crate::models::session::build_session(model_path, provider)?;

        Ok(Self {
            session: Mutex::new(session),
            score_thresh: 0.48,
            target_size: (480, 480),
        })
    }

    /// 对输入图像执行版面语义区域检测，返回原图绝对像素坐标
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<LayoutBox>, AnyOcrError> {
        let (orig_w, orig_h) = img.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return Ok(Vec::new());
        }

        let (tw, th) = self.target_size;

        // 1. 等比/缩放预处理 (使用 CatmullRom/Bicubic 插值保证高密表格细线特征不丢失)
        let resized = img.resize_exact(tw, th, FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        // 2. 归一化与 CHW 转换
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        let mut tensor_data = Array4::<f32>::zeros((1, 3, th as usize, tw as usize));
        for y in 0..th {
            for x in 0..tw {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    tensor_data[[0, c, y as usize, x as usize]] = (val - mean[c]) / std[c];
                }
            }
        }

        // 3. 构造 scale_factor [scale_y, scale_x]
        let scale_y = th as f32 / orig_h as f32;
        let scale_x = tw as f32 / orig_w as f32;
        let scale_factor_arr = ndarray::Array2::<f32>::from_shape_vec((1, 2), vec![scale_y, scale_x])
            .map_err(|e| AnyOcrError::InferenceError(format!("创建 scale_factor 失败: {e}")))?;

        // 4. ONNX 推理
        let mut session_guard = self
            .session
            .lock()
            .map_err(|_| AnyOcrError::InferenceError("获取 LayoutDetector Session 锁失败".into()))?;

        let input_img = Tensor::from_array(tensor_data)
            .map_err(|e| AnyOcrError::InferenceError(format!("转换图像 Tensor 失败: {e}")))?;
        let input_scale = Tensor::from_array(scale_factor_arr)
            .map_err(|e| AnyOcrError::InferenceError(format!("转换 scale_factor Tensor 失败: {e}")))?;

        let outputs = session_guard
            .run(ort::inputs![
                "image" => input_img,
                "scale_factor" => input_scale,
            ])
            .map_err(|e| AnyOcrError::InferenceError(format!("LayoutDetector 推理执行失败: {e}")))?;

        // 5. 解析输出 boxes: [N, 6]
        let (out_shape, out_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AnyOcrError::InferenceError(format!("提取检测输出 Tensor 失败: {e}")))?;

        let mut results = Vec::new();
        let rows = if out_shape.len() >= 2 { out_shape[0] as usize } else { 0 };
        let cols = if out_shape.len() >= 2 { out_shape[1] as usize } else { 6 };

        if cols >= 6 && out_data.len() >= rows * cols {
            for i in 0..rows {
                let offset = i * cols;
                let cls_id_f = out_data[offset];
                let cls_id = cls_id_f.round() as usize;
                let score = out_data[offset + 1];
                let mut x1 = out_data[offset + 2];
                let mut y1 = out_data[offset + 3];
                let mut x2 = out_data[offset + 4];
                let mut y2 = out_data[offset + 5];

                if score < self.score_thresh {
                    continue;
                }

                // 坐标安全 clamp
                x1 = x1.clamp(0.0, orig_w as f32);
                y1 = y1.clamp(0.0, orig_h as f32);
                x2 = x2.clamp(0.0, orig_w as f32);
                y2 = y2.clamp(0.0, orig_h as f32);

                let bw = x2 - x1;
                let bh = y2 - y1;
                if bw <= 1.0 || bh <= 1.0 {
                    continue;
                }

                let label = LayoutLabel::from_class_id(cls_id);

                // 过滤微小伪表格碎片 (表格物理尺寸与面积防噪保护)
                if label == LayoutLabel::Table && (bw < 60.0 || bh < 40.0 || (bw * bh) < 3000.0) {
                    continue;
                }

                results.push(LayoutBox {
                    label,
                    score,
                    x1,
                    y1,
                    x2,
                    y2,
                });
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_layout_detector_on_real_image() {
        let model_path = PathBuf::from("models/ocr/picodet_s_layout_17cls.onnx");
        if !model_path.exists() {
            println!("跳过测试：未找到模型文件 {}", model_path.display());
            return;
        }

        let detector = LayoutDetector::from_file(&model_path, crate::types::ExecutionProvider::Cpu)
            .expect("加载 LayoutDetector 失败");

        let img_path = PathBuf::from("/Users/icychick/Desktop/PDF/_page_images/Safety Jogger April 2026_p1.png");
        if !img_path.exists() {
            println!("跳过测试：未找到测试样本图 {}", img_path.display());
            return;
        }

        let img = image::open(&img_path).expect("读取测试样本图失败");
        let boxes = detector.detect(&img).expect("执行版面检测失败");

        println!("检出版面区域数量: {}", boxes.len());
        let mut found_table = false;
        for b in &boxes {
            println!("  -> {:?} (label: {}) score={:.3} coords=[{:.1}, {:.1}, {:.1}, {:.1}]", b.label, b.label.as_str(), b.score, b.x1, b.y1, b.x2, b.y2);
            if b.is_table() {
                found_table = true;
            }
        }
        assert!(found_table, "必须准确检出表格区域！");
    }
}
