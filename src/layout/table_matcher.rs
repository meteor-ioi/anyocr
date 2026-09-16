#[cfg(feature = "table")]
use crate::models::table::TableStructureResult;
#[cfg(feature = "table")]
use crate::types::DocBlock;
#[cfg(feature = "table")]
use std::collections::HashMap;
#[cfg(feature = "table")]
use crate::types::TextBoxItem;

/// 表格空间拓扑对齐与 GFM 表格构建器
pub struct TableMatcher;

impl TableMatcher {
    #[cfg(feature = "table")]
    /// 检验 SLANet 预测的表格骨架质量与可用性
    pub fn is_slanet_valid(table_res: &TableStructureResult, texts: &[TextBoxItem]) -> bool {
        if table_res.cell_boxes.is_empty() || table_res.html_tokens.is_empty() {
            return false;
        }

        // 统计落入各单元格的文本框数量与空间分布
        let mut cell_item_counts: HashMap<usize, usize> = HashMap::new();
        let mut cell_y_ranges: HashMap<usize, (f32, f32)> = HashMap::new();

        for item in texts {
            let cx = (item.coords[0] + item.coords[2]) / 2.0;
            let cy = (item.coords[1] + item.coords[3]) / 2.0;

            for cell in &table_res.cell_boxes {
                if cx >= cell.x1 && cx <= cell.x2 && cy >= cell.y1 && cy <= cell.y2 {
                    *cell_item_counts.entry(cell.cell_idx).or_insert(0) += 1;
                    let entry = cell_y_ranges.entry(cell.cell_idx).or_insert((cy, cy));
                    entry.0 = entry.0.min(cy);
                    entry.1 = entry.1.max(cy);
                    break;
                }
            }
        }

        // 1. 检查是否存在巨型异常单元格吞噬大量文本
        let total_text_count = texts.len();
        for (&_cell_idx, &count) in &cell_item_counts {
            // 单个单元格吞噬超过 15 个文本框且占整页文本框 35% 以上，说明发生了灾难性欠分割 (如畸形 rowspan="14")
            if count >= 15 && total_text_count >= 20 && (count as f32 / total_text_count as f32) > 0.35 {
                return false;
            }
        }

        // 2. 检查单元格利用率
        let total_cells = table_res.cell_boxes.len();
        let filled_cells = cell_item_counts.len();
        if total_cells >= 20 && filled_cells <= 3 {
            // 骨架预测严重失真
            return false;
        }

        true
    }

    #[cfg(feature = "table")]
    /// 将 OCR 文本与 SLANet 表格预测结果进行空间对齐与多块划分
    pub fn match_and_build(
        table_res: &TableStructureResult,
        texts: &[TextBoxItem],
    ) -> (Vec<TextBoxItem>, Option<DocBlock>, Vec<TextBoxItem>) {
        if !Self::is_slanet_valid(table_res, texts) {
            return (texts.to_vec(), None, Vec::new());
        }

        // 计算表格整体包围范围
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for cell in &table_res.cell_boxes {
            min_x = min_x.min(cell.x1);
            min_y = min_y.min(cell.y1);
            max_x = max_x.max(cell.x2);
            max_y = max_y.max(cell.y2);
        }

        let table_bbox = [min_x, min_y, max_x, max_y];
        let table_top = min_y - 5.0;
        let table_bottom = max_y + 5.0;

        let mut header_texts = Vec::new();
        let mut footer_texts = Vec::new();
        let mut in_table_texts = Vec::new();

        for item in texts {
            let cy = (item.coords[1] + item.coords[3]) / 2.0;
            let cx = (item.coords[0] + item.coords[2]) / 2.0;

            if cy < table_top {
                header_texts.push(item.clone());
            } else if cy > table_bottom {
                footer_texts.push(item.clone());
            } else if cx >= min_x - 10.0 && cx <= max_x + 10.0 {
                in_table_texts.push(item.clone());
            } else {
                header_texts.push(item.clone());
            }
        }

        // 空间反填：计算每个文本中心点落在哪个 CellBox 内
        let mut cell_texts: HashMap<usize, Vec<String>> = HashMap::new();
        for item in in_table_texts {
            let cx = (item.coords[0] + item.coords[2]) / 2.0;
            let cy = (item.coords[1] + item.coords[3]) / 2.0;

            let mut best_cell: Option<usize> = None;
            let mut min_dist = f32::INFINITY;

            for cell in &table_res.cell_boxes {
                if cx >= cell.x1 && cx <= cell.x2 && cy >= cell.y1 && cy <= cell.y2 {
                    best_cell = Some(cell.cell_idx);
                    break;
                }
                // 若中心点略微在边界外，计算到矩形中心的欧氏距离
                let ccx = (cell.x1 + cell.x2) / 2.0;
                let ccy = (cell.y1 + cell.y2) / 2.0;
                let dist = (cx - ccx).hypot(cy - ccy);
                if dist < min_dist {
                    min_dist = dist;
                    best_cell = Some(cell.cell_idx);
                }
            }

            if let Some(idx) = best_cell {
                cell_texts.entry(idx).or_default().push(item.text.trim().to_string());
            }
        }

        // 注入文字并生成干净 HTML / GFM 表格
        let mut aggregated_cells: HashMap<usize, String> = HashMap::new();
        for (idx, list) in cell_texts {
            aggregated_cells.insert(idx, list.join(" "));
        }

        let (gfm, raw_html) = Self::render_tokens_to_markdown(&table_res.html_tokens, &aggregated_cells);

        let table_block = DocBlock::Table {
            markdown_table: gfm,
            raw_html: Some(raw_html),
            bbox: table_bbox,
        };

        (header_texts, Some(table_block), footer_texts)
    }

    #[cfg(feature = "table")]
    fn render_tokens_to_markdown(
        tokens: &[String],
        cell_texts: &HashMap<usize, String>,
    ) -> (String, String) {
        let mut end_html = Vec::new();
        let mut td_index = 0usize;

        for tag in tokens {
            if !tag.contains("</td>") {
                end_html.push(tag.clone());
                continue;
            }

            if tag == "<td></td>" {
                end_html.push("<td>".to_string());
            }

            if let Some(text) = cell_texts.get(&td_index) {
                end_html.push(html_escape(text));
            }

            if tag == "<td></td>" {
                end_html.push("</td>".to_string());
            } else {
                end_html.push(tag.clone());
            }

            td_index += 1;
        }

        let filtered: Vec<String> = end_html
            .into_iter()
            .filter(|v| {
                v != "<thead>"
                    && v != "</thead>"
                    && v != "<tbody>"
                    && v != "</tbody>"
                    && v != "<html>"
                    && v != "</html>"
                    && v != "<body>"
                    && v != "</body>"
            })
            .collect();

        let table_body = filtered.join("");
        let table_html = if !table_body.trim().starts_with("<table") {
            format!("<table>{}</table>", table_body.trim())
        } else {
            table_body
        };

        let sanitized_table = table_html
            .replace("<td></td> rowspan=", "<td rowspan=")
            .replace("<td></td> colspan=", "<td colspan=");

        let gfm = try_convert_to_gfm(&sanitized_table).unwrap_or_default();
        (gfm, sanitized_table)
    }
}

#[cfg(feature = "table")]
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// 尝试将不含复杂合并的标准 HTML `<table>` 转换为 GFM 管道符表格
#[cfg(any(feature = "table", test))]
pub fn try_convert_to_gfm(html: &str) -> Option<String> {
    if html.contains("rowspan") || html.contains("colspan") {
        return None;
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut in_td = false;
    let mut current_cell = String::new();

    let mut chars = html.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '<' {
            let mut tag = String::new();
            while let Some(&nc) = chars.peek() {
                chars.next();
                if nc == '>' {
                    break;
                }
                tag.push(nc);
            }

            let tag_lower = tag.to_ascii_lowercase();
            if tag_lower == "tr" {
                current_row = Vec::new();
            } else if tag_lower == "/tr" {
                if !current_row.is_empty() {
                    rows.push(current_row.clone());
                }
            } else if tag_lower == "td" || tag_lower == "th" {
                in_td = true;
                current_cell = String::new();
            } else if tag_lower == "/td" || tag_lower == "/th" {
                in_td = false;
                current_row.push(current_cell.trim().replace('|', "\\|"));
            } else if in_td && (tag_lower == "br" || tag_lower.starts_with("br/")) {
                current_cell.push(' ');
            }
        } else if in_td {
            current_cell.push(c);
        }
    }

    if rows.is_empty() {
        return None;
    }

    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if max_cols == 0 {
        return None;
    }

    let mut out = String::new();
    for (r_idx, row) in rows.iter().enumerate() {
        let mut padded = row.clone();
        while padded.len() < max_cols {
            padded.push(String::new());
        }
        out.push_str(&format!("| {} |\n", padded.join(" | ")));

        // 首行作为表头后补充表头分割线
        if r_idx == 0 {
            let sep: Vec<&str> = vec!["---"; max_cols];
            out.push_str(&format!("| {} |\n", sep.join(" | ")));
        }
    }

    Some(out.trim_end().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_to_gfm_table() {
        let html = "<table><tr><td>列1</td><td>列2</td></tr><tr><td>A</td><td>B</td></tr></table>";
        let gfm = try_convert_to_gfm(html).unwrap();
        assert!(gfm.contains("| 列1 | 列2 |"));
        assert!(gfm.contains("| --- | --- |"));
        assert!(gfm.contains("| A | B |"));
    }
}
