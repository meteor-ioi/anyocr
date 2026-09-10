use crate::error::AnyOcrError;
use crate::ingestion::image::ImagePreprocessor;
use crate::types::TextBoxItem;
use image::{DynamicImage, GenericImageView};
use ort::session::Session;
use std::path::Path;
use std::sync::Mutex;

/// 基于 PP-OCRv6 的文本识别器 (SVTR / CRNN + CTC Greedy 解码)
pub struct TextRecognizer {
    session: Mutex<Session>,
    character_dict: Vec<String>,
}

impl TextRecognizer {
    /// 从 ONNX 模型和字符字典文件构造识别器
    pub fn from_files(
        model_path: impl AsRef<Path>,
        dict_path: impl AsRef<Path>,
    ) -> Result<Self, AnyOcrError> {
        let model_ref = model_path.as_ref();
        let dict_ref = dict_path.as_ref();

        if !model_ref.exists() {
            return Err(AnyOcrError::ModelNotReady(format!(
                "识别模型文件不存在: {}",
                model_ref.display()
            )));
        }
        if !dict_ref.exists() {
            return Err(AnyOcrError::ModelNotReady(format!(
                "字典文件不存在: {}",
                dict_ref.display()
            )));
        }

        // 读取字典字符映射
        let dict_content = std::fs::read_to_string(dict_ref)
            .map_err(AnyOcrError::IoError)?;

        let mut character_dict = Vec::new();
        // 第 0 项为空白字符 (Blank CTC Token)
        character_dict.push("".to_string());
        for line in dict_content.lines() {
            character_dict.push(line.to_string());
        }
        // 末尾追加空格字符，与 Paddle 规范保持一致
        character_dict.push(" ".to_string());

        let session = Session::builder()
            .map_err(|e| AnyOcrError::InferenceError(format!("创建 ONNX SessionBuilder 失败: {e}")))?
            .with_intra_threads(2)
            .map_err(|e| AnyOcrError::InferenceError(format!("配置推理线程失败: {e}")))?
            .commit_from_file(model_ref)
            .map_err(|e| AnyOcrError::InferenceError(format!("加载识别模型失败: {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            character_dict,
        })
    }

    /// 对单行裁剪切片执行文字识别与 CTC 贪心解码
    pub fn recognize_crop(&self, crop: &DynamicImage) -> Result<(String, f32), AnyOcrError> {
        let tensor = ImagePreprocessor::prepare_rec_input(crop, 48)?;

        let cow_array = tensor.into_dyn();
        let input_value = ort::value::Value::from_array(cow_array)
            .map_err(|e| AnyOcrError::InferenceError(format!("构建识别输入 Tensor 失败: {e}")))?;

        let mut session = self.session.lock().map_err(|e| {
            AnyOcrError::InferenceError(format!("获取识别模型锁失败: {e}"))
        })?;
        let outputs = session.run(ort::inputs![input_value])
            .map_err(|e| AnyOcrError::InferenceError(format!("执行识别推理失败: {e}")))?;

        let (_, output_value) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| AnyOcrError::InferenceError("识别模型未产生任何输出".to_string()))?;

        let (out_shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| AnyOcrError::InferenceError(format!("提取识别输出 Tensor 失败: {e}")))?;

        if out_shape.len() < 3 {
            return Err(AnyOcrError::InferenceError(format!(
                "识别输出维度异常: {:?}",
                out_shape
            )));
        }

        let time_steps = out_shape[1] as usize;
        let num_classes = out_shape[2] as usize;

        // CTC 贪心解码 (Greedy Search)
        let mut text = String::new();
        let mut score_sum = 0.0f32;
        let mut char_count = 0usize;
        let mut last_idx = 0usize;

        for t in 0..time_steps {
            let mut max_idx = 0usize;
            let mut max_prob = f32::MIN;
            let offset = t * num_classes;

            for c in 0..num_classes {
                let prob = data[offset + c];
                if prob > max_prob {
                    max_prob = prob;
                    max_idx = c;
                }
            }

            // CTC 规则：消除连续重复项与空白项 (0)
            if max_idx > 0 && max_idx != last_idx {
                if let Some(ch) = self.character_dict.get(max_idx) {
                    text.push_str(ch);
                    score_sum += max_prob;
                    char_count += 1;
                }
            }
            last_idx = max_idx;
        }

        let avg_score = if char_count > 0 {
            score_sum / char_count as f32
        } else {
            0.0
        };

        Ok((text, avg_score))
    }

    /// 批量识别文本行切片列表 (基于长宽比动态分桶批处理 Bucket Batching)
    pub fn recognize_batch(
        &self,
        crops: &[(DynamicImage, [f32; 4])],
    ) -> Vec<TextBoxItem> {
        if crops.is_empty() {
            return Vec::new();
        }

        if crops.len() == 1 {
            if let Ok((text, score)) = self.recognize_crop(&crops[0].0) {
                if !text.trim().is_empty() {
                    return vec![TextBoxItem {
                        text,
                        score,
                        coords: crops[0].1,
                    }];
                }
            }
            return Vec::new();
        }

        // 1. 计算每个切片的目标宽度并按宽高比划入 4 个桶
        // Bucket 0 (短文本):   <= 120px (~1-4 字符)
        // Bucket 1 (中等文本): 121..=320px (~5-12 字符)
        // Bucket 2 (长文本):   321..=640px (~13-25 字符)
        // Bucket 3 (超长文本): > 640px
        let mut buckets: [Vec<(usize, &DynamicImage, [f32; 4], u32)>; 4] = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];

        for (idx, (crop, bbox)) in crops.iter().enumerate() {
            let (w, h) = crop.dimensions();
            let ratio = 48.0 / h.max(1) as f32;
            let target_w = ((w as f32 * ratio) as u32).max(16);

            let bucket_idx = match target_w {
                0..=120 => 0,
                121..=320 => 1,
                321..=640 => 2,
                _ => 3,
            };

            buckets[bucket_idx].push((idx, crop, *bbox, target_w));
        }

        let mut all_results: Vec<(usize, TextBoxItem)> = Vec::with_capacity(crops.len());
        let batch_size = 8;

        // 2. 分桶批量前向推理
        for bucket in &buckets {
            for chunk in bucket.chunks(batch_size) {
                match self.recognize_chunk(chunk) {
                    Ok(batch_res) => {
                        all_results.extend(batch_res);
                    }
                    Err(e) => {
                        tracing::warn!("分桶批处理推理失败，自动平滑降级到逐行识别: {e}");
                        for &(orig_idx, crop, bbox, _) in chunk {
                            if let Ok((text, score)) = self.recognize_crop(crop) {
                                if !text.trim().is_empty() {
                                    all_results.push((orig_idx, TextBoxItem { text, score, coords: bbox }));
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. 恢复原始切片的几何排序顺序
        all_results.sort_by_key(|(orig_idx, _)| *orig_idx);
        all_results.into_iter().map(|(_, item)| item).collect()
    }

    /// 对单批同桶切片执行一次性 4D Tensor 前向批处理推理
    fn recognize_chunk(
        &self,
        chunk: &[(usize, &DynamicImage, [f32; 4], u32)],
    ) -> Result<Vec<(usize, TextBoxItem)>, AnyOcrError> {
        let b = chunk.len();
        if b == 0 {
            return Ok(Vec::new());
        }

        // 寻找当前 batch 内的最大宽度作为齐平宽度
        let max_w = chunk.iter().map(|item| item.3).max().unwrap_or(16).max(16);
        let mut tensor = ndarray::Array4::<f32>::from_elem((b, 3, 48, max_w as usize), -1.0);

        for (i, (_, crop, _, target_w)) in chunk.iter().enumerate() {
            let resized = crop.resize_exact(*target_w, 48, image::imageops::FilterType::Triangle);
            let rgb = resized.to_rgb8();

            for y in 0..48 {
                for x in 0..*target_w {
                    let pixel = rgb.get_pixel(x, y);
                    for c in 0..3 {
                        let val = pixel[c] as f32 / 255.0;
                        tensor[[i, c, y as usize, x as usize]] = (val - 0.5) / 0.5;
                    }
                }
            }
        }

        let cow_array = tensor.into_dyn();
        let input_value = ort::value::Value::from_array(cow_array)
            .map_err(|e| AnyOcrError::InferenceError(format!("构建分桶批处理 Tensor 失败: {e}")))?;

        let mut session = self.session.lock().map_err(|e| {
            AnyOcrError::InferenceError(format!("获取识别模型锁失败: {e}"))
        })?;

        let outputs = session.run(ort::inputs![input_value])
            .map_err(|e| AnyOcrError::InferenceError(format!("执行分桶批处理推理失败: {e}")))?;

        let (_, output_value) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| AnyOcrError::InferenceError("识别模型未产生任何输出".to_string()))?;

        let (out_shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| AnyOcrError::InferenceError(format!("提取识别批处理输出 Tensor 失败: {e}")))?;

        if out_shape.len() < 3 {
            return Err(AnyOcrError::InferenceError(format!("批处理输出维度异常: {:?}", out_shape)));
        }

        let time_steps = out_shape[1] as usize;
        let num_classes = out_shape[2] as usize;

        let mut chunk_results = Vec::with_capacity(b);

        for i in 0..b {
            let (orig_idx, _, bbox, _) = chunk[i];
            let batch_offset = i * time_steps * num_classes;

            let mut text = String::new();
            let mut score_sum = 0.0f32;
            let mut char_count = 0usize;
            let mut last_idx = 0usize;

            for t in 0..time_steps {
                let mut max_idx = 0usize;
                let mut max_prob = f32::MIN;
                let step_offset = batch_offset + t * num_classes;

                for c in 0..num_classes {
                    let prob = data[step_offset + c];
                    if prob > max_prob {
                        max_prob = prob;
                        max_idx = c;
                    }
                }

                if max_idx > 0 && max_idx != last_idx {
                    if let Some(ch) = self.character_dict.get(max_idx) {
                        text.push_str(ch);
                        score_sum += max_prob;
                        char_count += 1;
                    }
                }
                last_idx = max_idx;
            }

            let avg_score = if char_count > 0 {
                score_sum / char_count as f32
            } else {
                0.0
            };

            if !text.trim().is_empty() {
                chunk_results.push((orig_idx, TextBoxItem {
                    text,
                    score: avg_score,
                    coords: bbox,
                }));
            }
        }

        Ok(chunk_results)
    }
}
