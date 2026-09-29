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
    max_batch_size: usize,
}

impl TextRecognizer {
    /// 从 ONNX 模型和字符字典文件构造识别器 (支持指定硬件加速提供者与最大批大小)
    pub fn from_files(
        model_path: impl AsRef<Path>,
        dict_path: impl AsRef<Path>,
        provider: crate::types::ExecutionProvider,
        max_batch_size: usize,
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

        let session = crate::models::session::build_session(model_ref, provider)?;

        Ok(Self {
            session: Mutex::new(session),
            character_dict,
            max_batch_size: max_batch_size.clamp(1, 64),
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

        // 1. 计算每个切片的目标宽度 (保证在 [16, 960] 之间，防止畸形扁长图像产生超大无效 Tensor)
        let mut sorted_crops: Vec<(usize, &DynamicImage, [f32; 4], u32)> = Vec::with_capacity(crops.len());
        for (idx, (crop, bbox)) in crops.iter().enumerate() {
            let (w, h) = crop.dimensions();
            let ratio = 48.0 / h.max(1) as f32;
            let target_w = ((w as f32 * ratio).round() as u32).clamp(16, 960);
            sorted_crops.push((idx, crop, *bbox, target_w));
        }

        // 按 target_w 升序排序，使同一个 batch 内的切片宽度紧密贴合，彻底消除无效 padding 膨胀
        sorted_crops.sort_unstable_by_key(|item| item.3);

        let mut all_results: Vec<(usize, TextBoxItem)> = Vec::with_capacity(crops.len());
        let batch_size = self.max_batch_size;

        // 2. 紧凑批处理前向推理
        for chunk in sorted_crops.chunks(batch_size) {
            match self.recognize_chunk(chunk) {
                Ok(batch_res) => {
                    all_results.extend(batch_res);
                }
                Err(e) => {
                    tracing::warn!("批处理推理失败，自动平滑降级到逐行识别: {e}");
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

        // 3. 恢复原始切片的几何排序顺序
        all_results.sort_unstable_by_key(|(orig_idx, _)| *orig_idx);
        all_results.into_iter().map(|(_, item)| item).collect()
    }

    /// 对单批尺寸相近的切片执行一次性 4D Tensor 前向批处理推理
    fn recognize_chunk(
        &self,
        chunk: &[(usize, &DynamicImage, [f32; 4], u32)],
    ) -> Result<Vec<(usize, TextBoxItem)>, AnyOcrError> {
        let b = chunk.len();
        if b == 0 {
            return Ok(Vec::new());
        }

        // 寻找当前 batch 内的最大宽度作为齐平宽度 (经排序后当前 batch 宽高比极度紧凑)
        let max_w = chunk.iter().map(|item| item.3).max().unwrap_or(16).max(16) as usize;
        let mut tensor = ndarray::Array4::<f32>::from_elem((b, 3, 48, max_w), -1.0);
        let b_stride = 3 * 48 * max_w;
        let c_stride = 48 * max_w;

        if let Some(slice) = tensor.as_slice_mut() {
            for (i, (_, crop, _, target_w)) in chunk.iter().enumerate() {
                let tw = *target_w as usize;
                let rgb_crop = crop.to_rgb8();
                let resized = image::imageops::resize(&rgb_crop, *target_w, 48, image::imageops::FilterType::Triangle);
                let raw_bytes = resized.as_raw();
                let row_stride = tw * 3;
                let b_offset = i * b_stride;

                for y in 0..48 {
                    let src_row = &raw_bytes[y * row_stride..(y + 1) * row_stride];
                    let y_offset = y * max_w;
                    for (x, p) in src_row.chunks_exact(3).enumerate() {
                        let idx = y_offset + x;
                        slice[b_offset + idx] = p[0] as f32 / 127.5 - 1.0;
                        slice[b_offset + c_stride + idx] = p[1] as f32 / 127.5 - 1.0;
                        slice[b_offset + 2 * c_stride + idx] = p[2] as f32 / 127.5 - 1.0;
                    }
                }
            }
        } else {
            for (i, (_, crop, _, target_w)) in chunk.iter().enumerate() {
                let tw = *target_w as usize;
                let rgb_crop = crop.to_rgb8();
                let resized = image::imageops::resize(&rgb_crop, *target_w, 48, image::imageops::FilterType::Triangle);
                let raw_bytes = resized.as_raw();
                let stride = tw * 3;

                for y in 0..48 {
                    let src_row = &raw_bytes[y * stride..(y + 1) * stride];
                    for (x, p) in src_row.chunks_exact(3).enumerate() {
                        tensor[[i, 0, y, x]] = p[0] as f32 / 127.5 - 1.0;
                        tensor[[i, 1, y, x]] = p[1] as f32 / 127.5 - 1.0;
                        tensor[[i, 2, y, x]] = p[2] as f32 / 127.5 - 1.0;
                    }
                }
            }
        }

        let cow_array = tensor.into_dyn();
        let input_value = ort::value::Value::from_array(cow_array)
            .map_err(|e| AnyOcrError::InferenceError(format!("构建批处理 Tensor 失败: {e}")))?;

        let mut session = self.session.lock().map_err(|e| {
            AnyOcrError::InferenceError(format!("获取识别模型锁失败: {e}"))
        })?;

        let outputs = session.run(ort::inputs![input_value])
            .map_err(|e| AnyOcrError::InferenceError(format!("执行批处理推理失败: {e}")))?;

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
