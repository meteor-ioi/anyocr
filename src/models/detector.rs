use crate::error::AnyOcrError;
use crate::ingestion::image::{ImagePreprocessor, ResizeInfo};
use image::DynamicImage;
use ort::session::Session;
use std::path::Path;
use std::sync::Mutex;

/// 单个检测到的文本框 [x1, y1, x2, y2]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl DetBox {
    pub fn to_array(&self) -> [f32; 4] {
        [self.x1, self.y1, self.x2, self.y2]
    }

    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }
}

/// 基于 PP-OCRv6 的轻量高速文本定位器 (DBNet + RepLKFPN)
pub struct TextDetector {
    session: Mutex<Session>,
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
    max_side_len: u32,
}

impl TextDetector {
    /// 从 ONNX 模型文件构建检测器
    pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, AnyOcrError> {
        let path_ref = model_path.as_ref();
        if !path_ref.exists() {
            return Err(AnyOcrError::ModelNotReady(format!(
                "文本检测模型不存在: {}",
                path_ref.display()
            )));
        }

        let session = Session::builder()
            .map_err(|e| AnyOcrError::InferenceError(format!("创建 ONNX SessionBuilder 失败: {e}")))?
            .with_intra_threads(2)
            .map_err(|e| AnyOcrError::InferenceError(format!("配置推理线程失败: {e}")))?
            .commit_from_file(path_ref)
            .map_err(|e| AnyOcrError::InferenceError(format!("加载检测模型失败: {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            thresh: 0.3,
            box_thresh: 0.55,
            unclip_ratio: 1.5,
            max_side_len: 960,
        })
    }

    /// 执行单张图像的文本行定位
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<DetBox>, AnyOcrError> {
        let (tensor, resize_info) = ImagePreprocessor::prepare_det_input(img, self.max_side_len)?;

        // 构造 ONNX 输入 Tensor
        let cow_array = tensor.into_dyn();
        let input_value = ort::value::Value::from_array(cow_array)
            .map_err(|e| AnyOcrError::InferenceError(format!("构建输入 Tensor 失败: {e}")))?;

        // 推理执行
        let mut session = self.session.lock().map_err(|e| {
            AnyOcrError::InferenceError(format!("获取检测模型锁失败: {e}"))
        })?;
        let outputs = session.run(ort::inputs![input_value])
            .map_err(|e| AnyOcrError::InferenceError(format!("执行检测模型推理失败: {e}")))?;

        let (_, output_value) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| AnyOcrError::InferenceError("检测模型未产生任何输出".to_string()))?;

        let (out_shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| AnyOcrError::InferenceError(format!("提取输出概率图 Tensor 失败: {e}")))?;

        if out_shape.len() < 4 {
            return Err(AnyOcrError::InferenceError(format!(
                "检测输出维度异常: {:?}",
                out_shape
            )));
        }

        let h = out_shape[2] as usize;
        let w = out_shape[3] as usize;

        let prob_map = data[..h * w].to_vec();
        let boxes = self.post_process_prob_map(&prob_map, w, h, &resize_info);
        Ok(boxes)
    }

    /// DBNet 后处理：基于二值化连通区域标记提取文本框并膨胀反投影
    fn post_process_prob_map(
        &self,
        prob_map: &[f32],
        width: usize,
        height: usize,
        info: &ResizeInfo,
    ) -> Vec<DetBox> {
        // 1. 二值化掩膜
        let mut binary_mask = vec![false; width * height];
        for (i, &p) in prob_map.iter().enumerate() {
            if p >= self.thresh {
                binary_mask[i] = true;
            }
        }

        // 2. 连通块染色 (Connected Component Labeling)
        let mut labels = vec![0u32; width * height];
        let mut current_label = 0u32;
        let mut components: std::collections::HashMap<u32, (usize, usize, usize, usize, f32, u32)> =
            std::collections::HashMap::new();

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if binary_mask[idx] && labels[idx] == 0 {
                    current_label += 1;
                    // BFS 填充连通分量
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back((x, y));
                    labels[idx] = current_label;

                    let mut min_x = x;
                    let mut max_x = x;
                    let mut min_y = y;
                    let mut max_y = y;
                    let mut score_sum = prob_map[idx];
                    let mut point_count = 1u32;

                    while let Some((cx, cy)) = queue.pop_front() {
                        let neighbors = [
                            (cx.wrapping_sub(1), cy),
                            (cx + 1, cy),
                            (cx, cy.wrapping_sub(1)),
                            (cx, cy + 1),
                        ];

                        for (nx, ny) in neighbors {
                            if nx < width && ny < height {
                                let n_idx = ny * width + nx;
                                if binary_mask[n_idx] && labels[n_idx] == 0 {
                                    labels[n_idx] = current_label;
                                    queue.push_back((nx, ny));

                                    min_x = min_x.min(nx);
                                    max_x = max_x.max(nx);
                                    min_y = min_y.min(ny);
                                    max_y = max_y.max(ny);
                                    score_sum += prob_map[n_idx];
                                    point_count += 1;
                                }
                            }
                        }
                    }

                    components.insert(
                        current_label,
                        (min_x, min_y, max_x, max_y, score_sum, point_count),
                    );
                }
            }
        }

        // 3. 过滤过小面积或低置信度的框，并执行膨胀与坐标映射
        let mut det_boxes = Vec::new();

        for (_, (min_x, min_y, max_x, max_y, score_sum, count)) in components {
            let bw = (max_x - min_x + 1) as f32;
            let bh = (max_y - min_y + 1) as f32;

            if bw < 4.0 || bh < 4.0 {
                continue;
            }

            let avg_score = score_sum / count as f32;
            if avg_score < self.box_thresh {
                continue;
            }

            // Unclip 膨胀处理
            let expand_x = (bw * (self.unclip_ratio - 1.0) / 2.0).max(1.0);
            let expand_y = (bh * (self.unclip_ratio - 1.0) / 2.0).max(1.0);

            let unclipped_x1 = (min_x as f32 - expand_x).max(0.0);
            let unclipped_y1 = (min_y as f32 - expand_y).max(0.0);
            let unclipped_x2 = (max_x as f32 + expand_x).min(width as f32);
            let unclipped_y2 = (max_y as f32 + expand_y).min(height as f32);

            // 逆变换到原图物理像素坐标
            let orig_x1 = (unclipped_x1 / info.scale_x).min(info.original_width as f32);
            let orig_y1 = (unclipped_y1 / info.scale_y).min(info.original_height as f32);
            let orig_x2 = (unclipped_x2 / info.scale_x).min(info.original_width as f32);
            let orig_y2 = (unclipped_y2 / info.scale_y).min(info.original_height as f32);

            det_boxes.push(DetBox {
                x1: orig_x1,
                y1: orig_y1,
                x2: orig_x2,
                y2: orig_y2,
                score: avg_score,
            });
        }

        // 4. 自上而下、自左向右排序 (Reading Order, 全序排序避免 NaN panic)
        det_boxes.sort_by(|a, b| {
            a.y1.total_cmp(&b.y1).then_with(|| a.x1.total_cmp(&b.x1))
        });

        det_boxes
    }
}
