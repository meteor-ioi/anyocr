#[cfg(feature = "ofd")]
use crate::engine::Engine;
use crate::error::AnyOcrError;
use crate::ingestion::image::ImagePreprocessor;
use crate::layout::LayoutEngine;
use crate::types::{PageResult, ParsedDocument, TextBoxItem};
use std::io::{Cursor, Read};
use std::time::Instant;

/// 国标版式文档 (OFD, GB/T 33190-2016) 摄入与解析器
pub struct OfdIngestion;

impl OfdIngestion {
    #[cfg(feature = "ofd")]
    /// 解析 OFD 容器字节流为多页 ParsedDocument
    pub fn parse_ofd(bytes: &[u8], engine: &Engine) -> Result<ParsedDocument, AnyOcrError> {
        let t0 = Instant::now();

        let reader = Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(reader)
            .map_err(|e| AnyOcrError::OfdError(format!("解压 OFD 容器失败: {e}")))?;

        // 1. 检索所有页面 Content.xml 路径清单
        let mut page_xml_names: Vec<String> = Vec::new();
        for i in 0..archive.len() {
            if let Ok(file) = archive.by_index(i) {
                let name = file.name();
                if name.ends_with("Content.xml") || (name.contains("Page_") && name.ends_with(".xml")) {
                    page_xml_names.push(name.to_string());
                }
            }
        }

        // 页面按自然序排序 (如 Page_0, Page_1 ...)
        page_xml_names.sort();

        let mut page_results: Vec<PageResult> = Vec::new();
        let mut all_markdown_sections: Vec<String> = Vec::new();

        for (page_idx, xml_name) in page_xml_names.iter().enumerate() {
            let mut xml_content = String::new();
            if let Ok(mut file) = archive.by_name(xml_name) {
                let _ = file.read_to_string(&mut xml_content);
            }

            let extracted_items = Self::extract_text_codes_from_xml(&xml_content);

            if !extracted_items.is_empty() {
                // 原生文字型页面：直接通过版面重构引擎还原
                let (blocks, md) = LayoutEngine::process(&extracted_items, (595, 842), None);
                all_markdown_sections.push(md);
                page_results.push(PageResult {
                    page_index: page_idx,
                    dimensions: (595, 842),
                    blocks,
                    boxes: extracted_items,
                });
            } else {
                // 页面可能为纯图片型电子公文或发票，检索关联内嵌图
                let mut page_image = None;
                for i in 0..archive.len() {
                    if let Ok(mut file) = archive.by_index(i) {
                        let name = file.name().to_ascii_lowercase();
                        if (name.ends_with(".jpg") || name.ends_with(".png") || name.ends_with(".jpeg"))
                            && (name.contains("res") || name.contains("image"))
                        {
                            let mut buf = Vec::new();
                            if file.read_to_end(&mut buf).is_ok() {
                                if let Ok(dyn_img) = ImagePreprocessor::decode_image(&buf) {
                                    page_image = Some(dyn_img);
                                    break;
                                }
                            }
                        }
                    }
                }

                if let Some(img) = page_image {
                    let mut parsed_page = engine.parse_image(&img)?;
                    if let Some(mut p) = parsed_page.pages.pop() {
                        p.page_index = page_idx;
                        all_markdown_sections.push(parsed_page.markdown);
                        page_results.push(p);
                    }
                } else {
                    return Err(AnyOcrError::OfdError(format!(
                        "第 {page_idx} 页未检测到有效可编辑文字或内嵌位图，当前纯 Rust 引擎不支持纯矢量光栅化"
                    )));
                }
            }
        }

        // 若页面 XML 为空，尝试直接在 Res 目录下检索图片作为单据提取
        if page_results.is_empty() {
            let mut fallback_image = None;
            for i in 0..archive.len() {
                if let Ok(mut file) = archive.by_index(i) {
                    let name = file.name().to_ascii_lowercase();
                    if name.ends_with(".jpg") || name.ends_with(".png") || name.ends_with(".jpeg") {
                        let mut buf = Vec::new();
                        if file.read_to_end(&mut buf).is_ok() {
                            if let Ok(dyn_img) = ImagePreprocessor::decode_image(&buf) {
                                fallback_image = Some(dyn_img);
                                break;
                            }
                        }
                    }
                }
            }

            if let Some(img) = fallback_image {
                let parsed_page = engine.parse_image(&img)?;
                return Ok(parsed_page);
            }

            return Err(AnyOcrError::OfdError("OFD 容器内未找到任何可识别页面或图像".to_string()));
        }

        let elapsed_ms = t0.elapsed().as_millis() as u64;
        let total_pages = page_results.len();
        let markdown = all_markdown_sections.join("\n\n---\n\n");

        Ok(ParsedDocument {
            markdown,
            pages: page_results,
            total_pages,
            elapsed_ms,
        })
    }

    /// 从 OFD 页面 XML 中快速提取 <ofd:TextCode> 文本与其坐标
    pub fn extract_text_codes_from_xml(xml: &str) -> Vec<TextBoxItem> {
        let mut reader = quick_xml::reader::Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut items = Vec::new();
        let mut current_coords = [20.0f32, 20.0, 300.0, 40.0];
        let mut in_text_code = false;

        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(quick_xml::events::Event::Start(ref e)) => {
                    let name = e.name();
                    if name.as_ref() == b"ofd:TextCode" || name.as_ref() == b"TextCode" {
                        in_text_code = true;
                        // 提取 X, Y 坐标
                        let mut x = 20.0f32;
                        let mut y = 20.0f32;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"X" {
                                if let Ok(val_str) = std::str::from_utf8(&attr.value) {
                                    x = val_str.parse().unwrap_or(20.0);
                                }
                            } else if attr.key.as_ref() == b"Y" {
                                if let Ok(val_str) = std::str::from_utf8(&attr.value) {
                                    y = val_str.parse().unwrap_or(20.0);
                                }
                            }
                        }
                        current_coords = [x, y, x + 200.0, y + 20.0];
                    }
                }
                Ok(quick_xml::events::Event::Text(ref e)) => {
                    if in_text_code {
                        if let Ok(text) = std::str::from_utf8(e.as_ref()) {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                items.push(TextBoxItem {
                                    text: trimmed.to_string(),
                                    score: 1.0,
                                    coords: current_coords,
                                });
                            }
                        }
                    }
                }
                Ok(quick_xml::events::Event::End(ref e)) => {
                    let name = e.name();
                    if name.as_ref() == b"ofd:TextCode" || name.as_ref() == b"TextCode" {
                        in_text_code = false;
                    }
                }
                Ok(quick_xml::events::Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
            buf.clear();
        }

        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ofd_text_code() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <ofd:Page xmlns:ofd="http://www.ofdspec.org/2016">
            <ofd:Content>
                <ofd:Layer>
                    <ofd:TextObject Boundary="20 30 200 40">
                        <ofd:TextCode X="25.0" Y="35.0">增值税专用发票</ofd:TextCode>
                    </ofd:TextObject>
                </ofd:Layer>
            </ofd:Content>
        </ofd:Page>"#;

        let items = OfdIngestion::extract_text_codes_from_xml(xml);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "增值税专用发票");
        assert_eq!(items[0].coords[0], 25.0);
        assert_eq!(items[0].coords[1], 35.0);
    }
}
