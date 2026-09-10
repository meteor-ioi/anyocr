use crate::error::AnyOcrError;
use crate::ingestion::image::ImagePreprocessor;
use image::DynamicImage;
use ort::session::Session;
use std::path::Path;
use std::sync::Mutex;

/// 预测出的单个单元格物理坐标框
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellBox {
    pub cell_idx: usize,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// 表格拓扑与骨架解析结果
#[derive(Debug, Clone)]
pub struct TableStructureResult {
    /// HTML 骨架标签 Token 序列 (如 ["<html>", "<body>", "<table>", "<tr>", "<td></td>", ...] )
    pub html_tokens: Vec<String>,
    /// 预测的所有单元格在原图上的物理坐标框
    pub cell_boxes: Vec<CellBox>,
    /// 每个 tr 行对应的单元格索引清单
    pub rows: Vec<Vec<usize>>,
}

impl TableStructureResult {
    /// 获取核心多列表格的 (起始行索引, 结束行索引)（含首尾）
    pub fn core_table_row_range(&self) -> Option<(usize, usize)> {
        let mut first = None;
        let mut last = None;
        for (r_idx, row) in self.rows.iter().enumerate() {
            if row.len() >= 2 {
                if first.is_none() {
                    first = Some(r_idx);
                }
                last = Some(r_idx);
            }
        }
        match (first, last) {
            (Some(f), Some(l)) => Some((f, l)),
            _ => None,
        }
    }

    /// 提取仅包含核心多列表格的 html_tokens 片段
    pub fn core_table_tokens(&self) -> Vec<String> {
        if let Some((start_r, end_r)) = self.core_table_row_range() {
            let mut result = Vec::new();
            let mut current_r = 0usize;
            let mut in_target = false;
            let mut in_row = false;

            for tok in &self.html_tokens {
                if tok == "<tr>" {
                    in_row = true;
                    if current_r >= start_r && current_r <= end_r {
                        in_target = true;
                    } else {
                        in_target = false;
                    }
                }

                if in_target {
                    result.push(tok.clone());
                }

                if tok == "</tr>" {
                    if in_row {
                        current_r += 1;
                        in_row = false;
                    }
                    if current_r > end_r {
                        in_target = false;
                    }
                }
            }
            result
        } else {
            Vec::new()
        }
    }
}

/// 基于 SLANet_plus 的表格结构预测器
pub struct TableStructurePredictor {
    session: Mutex<Session>,
    vocab: Vec<String>,
}

impl TableStructurePredictor {
    /// 从 ONNX 模型文件构建预测器并初始化内置标准词表
    pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, AnyOcrError> {
        let path_ref = model_path.as_ref();
        if !path_ref.exists() {
            return Err(AnyOcrError::ModelNotReady(format!(
                "表格模型文件不存在: {}",
                path_ref.display()
            )));
        }

        let session = Session::builder()
            .map_err(|e| AnyOcrError::InferenceError(format!("创建 ONNX SessionBuilder 失败: {e}")))?
            .with_intra_threads(2)
            .map_err(|e| AnyOcrError::InferenceError(format!("配置推理线程失败: {e}")))?
            .commit_from_file(path_ref)
            .map_err(|e| AnyOcrError::InferenceError(format!("加载表格模型失败: {e}")))?;

        // SLANet_plus 官方标准 50 词元结构词表
        let mut vocab = Vec::with_capacity(50);
        vocab.push("sos".to_string());
        vocab.push("<thead>".to_string());
        vocab.push("</thead>".to_string());
        vocab.push("<tbody>".to_string());
        vocab.push("</tbody>".to_string());
        vocab.push("<tr>".to_string());
        vocab.push("</tr>".to_string());
        vocab.push("<td".to_string());
        vocab.push(">".to_string());
        vocab.push("</td>".to_string());
        for i in 2..=20 {
            vocab.push(format!(" colspan=\"{i}\""));
        }
        for i in 2..=20 {
            vocab.push(format!(" rowspan=\"{i}\""));
        }
        vocab.push("<td></td>".to_string());
        vocab.push("eos".to_string());

        Ok(Self {
            session: Mutex::new(session),
            vocab,
        })
    }

    /// 执行单张表格图像的拓扑骨架预测
    pub fn predict(&self, img: &DynamicImage) -> Result<TableStructureResult, AnyOcrError> {
        let (tensor, resize_info) = ImagePreprocessor::prepare_table_input(img)?;

        let cow_array = tensor.into_dyn();
        let input_value = ort::value::Value::from_array(cow_array)
            .map_err(|e| AnyOcrError::InferenceError(format!("构建表格输入 Tensor 失败: {e}")))?;

        let mut session = self.session.lock().map_err(|e| {
            AnyOcrError::InferenceError(format!("获取表格模型锁失败: {e}"))
        })?;
        let outputs = session.run(ort::inputs![input_value])
            .map_err(|e| AnyOcrError::InferenceError(format!("执行表格模型推理失败: {e}")))?;

        let mut out_iter = outputs.into_iter();
        let (_, val1) = out_iter
            .next()
            .ok_or_else(|| AnyOcrError::InferenceError("表格模型输出数量小于 1".to_string()))?;
        let (_, val2) = out_iter
            .next()
            .ok_or_else(|| AnyOcrError::InferenceError("表格模型输出数量小于 2".to_string()))?;

        let (shape1, data1) = val1
            .try_extract_tensor::<f32>()
            .map_err(|e| AnyOcrError::InferenceError(format!("提取表格输出 1 Tensor 失败: {e}")))?;
        let (shape2, data2) = val2
            .try_extract_tensor::<f32>()
            .map_err(|e| AnyOcrError::InferenceError(format!("提取表格输出 2 Tensor 失败: {e}")))?;

        // 识别 structure probs 与 bbox loc
        let (s_shape, s_data, b_shape, b_data) = if shape1.len() >= 3 && (shape1[2] == 8 || shape1[2] == 4) {
            (shape2, data2, shape1, data1)
        } else if shape2.len() >= 3 && (shape2[2] == 8 || shape2[2] == 4) {
            (shape1, data1, shape2, data2)
        } else if shape1.len() >= 3 && shape1[2] == 50 {
            (shape1, data1, shape2, data2)
        } else {
            (shape2, data2, shape1, data1)
        };

        if s_shape.len() < 3 || b_shape.len() < 3 {
            return Err(AnyOcrError::InferenceError("表格输出 Tensor 维度不匹配".to_string()));
        }

        let steps = s_shape[1] as usize;
        let vocab_size = s_shape[2] as usize;
        let b_dim = b_shape[2] as usize;

        let mut html_tokens = Vec::new();
        let mut cell_boxes = Vec::new();
        let mut cell_idx = 0usize;
        let mut current_row_cell_indices = Vec::new();
        let mut rows: Vec<Vec<usize>> = Vec::new();

        let max_dim = (resize_info.original_width as f32).max(resize_info.original_height as f32);

        for step in 0..steps {
            let mut max_idx = 0usize;
            let mut max_prob = f32::MIN;
            let s_offset = step * vocab_size;

            for v in 0..vocab_size {
                let prob = s_data[s_offset + v];
                if prob > max_prob {
                    max_prob = prob;
                    max_idx = v;
                }
            }

            let token = self.vocab.get(max_idx).cloned().unwrap_or_default();
            if token == "eos" || token.is_empty() {
                break;
            }
            if token == "sos" {
                continue;
            }

            if token == "<tr>" {
                current_row_cell_indices.clear();
            } else if token == "</tr>" {
                if !current_row_cell_indices.is_empty() {
                    rows.push(current_row_cell_indices.clone());
                    current_row_cell_indices.clear();
                }
            }

            // 若 token 代表单元格开始，提取其在原图上的 bbox
            if (token == "<td></td>" || token == "<td>" || token == "<td") && b_dim >= 4 {
                let b_offset = step * b_dim;
                let (norm_x1, norm_y1, norm_x2, norm_y2) = if b_dim >= 8 {
                    let x0 = b_data[b_offset];
                    let y0 = b_data[b_offset + 1];
                    let x1 = b_data[b_offset + 2];
                    let y1 = b_data[b_offset + 3];
                    let x2 = b_data[b_offset + 4];
                    let y2 = b_data[b_offset + 5];
                    let x3 = b_data[b_offset + 6];
                    let y3 = b_data[b_offset + 7];

                    let min_x = x0.min(x1).min(x2).min(x3).max(0.0).min(1.0);
                    let min_y = y0.min(y1).min(y2).min(y3).max(0.0).min(1.0);
                    let max_x = x0.max(x1).max(x2).max(x3).max(0.0).min(1.0);
                    let max_y = y0.max(y1).max(y2).max(y3).max(0.0).min(1.0);
                    (min_x, min_y, max_x, max_y)
                } else {
                    let min_x = b_data[b_offset].max(0.0).min(1.0);
                    let min_y = b_data[b_offset + 1].max(0.0).min(1.0);
                    let max_x = b_data[b_offset + 2].max(0.0).min(1.0);
                    let max_y = b_data[b_offset + 3].max(0.0).min(1.0);
                    (min_x, min_y, max_x, max_y)
                };

                let real_x1 = (norm_x1 * max_dim).max(0.0).min(resize_info.original_width as f32);
                let real_y1 = (norm_y1 * max_dim).max(0.0).min(resize_info.original_height as f32);
                let real_x2 = (norm_x2 * max_dim).max(0.0).min(resize_info.original_width as f32);
                let real_y2 = (norm_y2 * max_dim).max(0.0).min(resize_info.original_height as f32);

                cell_boxes.push(CellBox {
                    cell_idx,
                    x1: real_x1,
                    y1: real_y1,
                    x2: real_x2,
                    y2: real_y2,
                });
                current_row_cell_indices.push(cell_idx);
                cell_idx += 1;
            }

            html_tokens.push(token);
        }

        Ok(TableStructureResult {
            html_tokens,
            cell_boxes,
            rows,
        })
    }
}
