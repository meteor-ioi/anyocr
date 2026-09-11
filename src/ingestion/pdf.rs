#[cfg(feature = "pdf")]
use crate::engine::Engine;
use crate::error::AnyOcrError;
use crate::ingestion::image::ImagePreprocessor;
use crate::layout::LayoutEngine;
use crate::types::{PageResult, ParsedDocument, TextBoxItem};
#[cfg(feature = "pdf")]
use lopdf::Document;
use std::time::Instant;

/// 页面并行预处理产物
enum PreparedPage {
    /// 健全的原生可编辑文本
    Text(String),
    /// 已解码且旋转校正完成的图像
    Image(image::DynamicImage),
    /// 纯矢量页面无图片
    VectorOnly(u32),
}

/// PDF 文档摄入与解析器 (支持双层可编辑 PDF、乱码坏死层熔断回退与纯扫描件 OCR)
pub struct PdfIngestion;

impl PdfIngestion {
    #[cfg(feature = "pdf")]
    /// 准备单页输入：并行执行可编辑文本提取、内嵌主图提取、格式解码与旋转校正
    fn prepare_page(
        doc: &Document,
        _page_idx: usize,
        page_num: u32,
        page_id: lopdf::ObjectId,
    ) -> PreparedPage {
        let extracted_text = doc.extract_text(&[page_num]).unwrap_or_default();
        let is_dead_layer = Self::is_dead_or_corrupted_text(&extracted_text);

        if !extracted_text.trim().is_empty() && !is_dead_layer {
            return PreparedPage::Text(extracted_text);
        }

        // 提取内嵌主图
        let mut best_img: Option<image::DynamicImage> = None;
        if let Ok(images) = doc.get_page_images(page_id) {
            let mut max_pixels = 0u64;
            for img_obj in images {
                if let Ok(dyn_img) = ImagePreprocessor::decode_image(&img_obj.content) {
                    let pixels = (dyn_img.width() as u64) * (dyn_img.height() as u64);
                    if pixels > max_pixels {
                        max_pixels = pixels;
                        best_img = Some(dyn_img);
                    }
                }
            }
        }

        if let Some(mut img) = best_img {
            let rotation = Self::get_page_rotation(doc, page_id);
            img = match (rotation % 360 + 360) % 360 {
                90 => img.rotate90(),
                180 => img.rotate180(),
                270 => img.rotate270(),
                _ => img,
            };
            PreparedPage::Image(img)
        } else {
            PreparedPage::VectorOnly(page_num)
        }
    }

    #[cfg(feature = "pdf")]
    /// 解析 PDF 字节流为多页 ParsedDocument
    pub fn parse_pdf(bytes: &[u8], engine: &Engine) -> Result<ParsedDocument, AnyOcrError> {
        let t0 = Instant::now();

        let doc = Document::load_mem(bytes)
            .map_err(|e| AnyOcrError::PdfError(format!("加载 PDF 文档失败: {e}")))?;

        let pages = doc.get_pages();
        if pages.is_empty() {
            return Err(AnyOcrError::PdfError("PDF 文档未包含有效页面".to_string()));
        }

        let mut page_results = Vec::with_capacity(pages.len());
        let mut all_markdown_sections = Vec::with_capacity(pages.len());

        let page_entries: Vec<(usize, u32, lopdf::ObjectId)> = pages
            .iter()
            .enumerate()
            .map(|(idx, (&num, &id))| (idx, num, id))
            .collect();

        // 1. 页面预处理流水线：多页并发提取文本、解码扫描图并做旋转校正 (消除串行解码阻塞)
        let prepared_pages: Vec<PreparedPage> = if page_entries.len() <= 1 {
            page_entries
                .into_iter()
                .map(|(idx, num, id)| Self::prepare_page(&doc, idx, num, id))
                .collect()
        } else {
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(page_entries.len());
                for (idx, num, id) in page_entries {
                    let d = &doc;
                    handles.push(s.spawn(move || Self::prepare_page(d, idx, num, id)));
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };

        // 2. 顺次执行 OCR 与版面排版推理，严格保持页序与 Session 并发安全性
        for (page_idx, prepared) in prepared_pages.into_iter().enumerate() {
            match prepared {
                PreparedPage::Text(extracted_text) => {
                    let page_res = Self::build_page_from_text(&extracted_text, page_idx);
                    all_markdown_sections.push(page_res.blocks.iter().map(|b| b.to_markdown()).collect::<Vec<_>>().join("\n\n"));
                    page_results.push(page_res);
                }
                PreparedPage::Image(img) => {
                    let mut parsed_page = engine.parse_image(&img)?;
                    if let Some(mut p) = parsed_page.pages.pop() {
                        p.page_index = page_idx;
                        all_markdown_sections.push(parsed_page.markdown);
                        page_results.push(p);
                    }
                }
                PreparedPage::VectorOnly(page_num) => {
                    return Err(AnyOcrError::PdfError(format!(
                        "第 {page_num} 页为纯矢量绘图且无内嵌位图，当前零外部 C 动态库基线不支持光栅化渲染，请确保 PDF 包含扫描图或文本层"
                    )));
                }
            }
        }

        let elapsed_ms = t0.elapsed().as_millis() as u64;
        let total_pages = page_results.len();
        // 多页文档之间以标准 Markdown 分页线分隔
        let markdown = all_markdown_sections.join("\n\n---\n\n");

        Ok(ParsedDocument {
            markdown,
            pages: page_results,
            total_pages,
            elapsed_ms,
        })
    }

    /// 检测抽取的文本是否为乱码或损坏的 CMap 字体坏死层
    /// 统计不可见控制字符、Unicode 替换字符 \u{FFFD} 及非法乱码的比例
    pub fn is_dead_or_corrupted_text(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return true;
        }

        let total_chars = trimmed.chars().count();
        if total_chars == 0 {
            return true;
        }

        let mut bad_chars = 0;
        for c in trimmed.chars() {
            if c == '\u{FFFD}' {
                bad_chars += 1;
            } else if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
                bad_chars += 1;
            } else if c == '\0' {
                bad_chars += 1;
            }
        }

        let bad_ratio = bad_chars as f32 / total_chars as f32;
        bad_ratio >= 0.25
    }

    #[cfg(feature = "pdf")]
    /// 从纯文本内容构建 PageResult
    fn build_page_from_text(text: &str, page_idx: usize) -> PageResult {
        let lines: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
        let mut items = Vec::with_capacity(lines.len());

        let mut y = 20.0f32;
        for l in lines {
            items.push(TextBoxItem {
                text: l.to_string(),
                score: 1.0,
                coords: [20.0, y, 500.0, y + 20.0],
            });
            y += 26.0;
        }

        let (blocks, _) = LayoutEngine::process(&items, (595, 842), None);

        PageResult {
            page_index: page_idx,
            dimensions: (595, 842), // 标准 A4 典型点阵宽高
            blocks,
            boxes: items,
        }
    }

    #[cfg(feature = "pdf")]
    /// 获取指定 PDF 页面的顺时针旋转角度 (优先页面自身 Rotate，次之递归回溯 Parent)
    fn get_page_rotation(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> i64 {
        let mut curr_id = page_id;
        for _ in 0..10 {
            if let Ok(obj) = doc.get_object(curr_id) {
                if let Ok(dict) = obj.as_dict() {
                    if let Ok(rot_obj) = dict.get(b"Rotate") {
                        if let Ok(r) = rot_obj.as_i64() {
                            return r;
                        }
                    }
                    if let Ok(parent_obj) = dict.get(b"Parent") {
                        if let Ok(p_id) = parent_obj.as_reference() {
                            curr_id = p_id;
                            continue;
                        }
                    }
                }
            }
            break;
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dead_text_detection() {
        assert!(PdfIngestion::is_dead_or_corrupted_text(""));
        assert!(PdfIngestion::is_dead_or_corrupted_text("\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}"));
        assert!(!PdfIngestion::is_dead_or_corrupted_text("这是正常的 PDF 文本内容。"));
    }
}
