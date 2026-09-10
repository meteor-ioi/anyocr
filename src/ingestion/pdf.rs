#[cfg(feature = "pdf")]
use crate::engine::Engine;
use crate::error::AnyOcrError;
use crate::ingestion::image::ImagePreprocessor;
use crate::layout::LayoutEngine;
use crate::types::{PageResult, ParsedDocument, TextBoxItem};
#[cfg(feature = "pdf")]
use lopdf::Document;
use std::time::Instant;

/// PDF 文档摄入与解析器 (支持双层可编辑 PDF、乱码坏死层熔断回退与纯扫描件 OCR)
pub struct PdfIngestion;

impl PdfIngestion {
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

        for (page_idx, (&page_num, &_page_id)) in pages.iter().enumerate() {
            // 1. 尝试直接抽取原生文字流
            let extracted_text = doc.extract_text(&[page_num]).unwrap_or_default();
            let is_dead_layer = Self::is_dead_or_corrupted_text(&extracted_text);

            if !extracted_text.trim().is_empty() && !is_dead_layer {
                // 原生可编辑文字层健全：直接按行解析构建 PageResult
                let page_res = Self::build_page_from_text(&extracted_text, page_idx);
                all_markdown_sections.push(page_res.blocks.iter().map(|b| b.to_markdown()).collect::<Vec<_>>().join("\n\n"));
                page_results.push(page_res);
            } else {
                // 2. 文字层为空或坏死 (CMap 损坏)，自动熔断回退至页面内嵌图像提取并走 OCR
                let mut page_image: Option<image::DynamicImage> = None;

                if let Ok(images) = doc.get_page_images(_page_id) {
                    // 优先选择面积最大的主扫描图
                    let mut best_img: Option<image::DynamicImage> = None;
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
                    page_image = best_img;
                }

                if let Some(img) = page_image {
                    let mut parsed_page = engine.parse_image(&img)?;
                    if let Some(mut p) = parsed_page.pages.pop() {
                        p.page_index = page_idx;
                        all_markdown_sections.push(parsed_page.markdown);
                        page_results.push(p);
                    }
                } else {
                    // 3. 既无文字，也无内嵌栅格图（纯矢量绘制页面）
                    // 遵守 Systems Architecture Guardrails：显式失败，严禁静默吞掉
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
