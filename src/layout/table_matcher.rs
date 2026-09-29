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
    /// 针对已由版面模型确定的局部表格 ROI 及其对应的 SLANet 预测结果，生成高保真 DocBlock::Table
    pub fn build_table_from_roi(
        table_res: &TableStructureResult,
        in_table_texts: &[TextBoxItem],
        roi_bbox: [f32; 4],
    ) -> Option<DocBlock> {
        if table_res.cell_boxes.is_empty() || table_res.html_tokens.is_empty() {
            return None;
        }

        // 使用二维重叠面积 (IoF) 与空间拓扑排序分配文本至单元格
        let aggregated_cells = Self::assign_texts_to_cells(&table_res.cell_boxes, in_table_texts);
        let (gfm, raw_html) = Self::render_tokens_to_markdown(&table_res.html_tokens, &aggregated_cells);

        Some(DocBlock::Table {
            markdown_table: gfm,
            raw_html: Some(raw_html),
            bbox: roi_bbox,
        })
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

        // 空间反填：使用二维重叠面积 (IoF) 与空间拓扑排序分配文本至单元格
        let aggregated_cells = Self::assign_texts_to_cells(&table_res.cell_boxes, &in_table_texts);
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

        let raw_table = table_html
            .replace("<td></td> rowspan=", "<td rowspan=")
            .replace("<td></td> colspan=", "<td colspan=");

        // 自动清除 HTML 表格中没有任何文字内容的纯空行 (如 <tr><td></td></tr>、<tr><td colspan="2"></td></tr>)
        use std::sync::LazyLock;
        static TR_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(r"(?is)<tr\b[^>]*>.*?</tr>").expect("合法正则")
        });
        static TAG_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(r"<[^>]+>").expect("合法正则")
        });
        static TABLE_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(r"(?is)<table\b[^>]*>.*?</table>").expect("合法正则")
        });

        let no_empty_trs = TR_RE.replace_all(&raw_table, |caps: &regex::Captures| {
            let tr_str = &caps[0];
            let stripped = TAG_RE.replace_all(tr_str, "");
            let clean = stripped.replace("&nbsp;", " ").replace("&#160;", " ").trim().to_string();
            if clean.is_empty() {
                String::new()
            } else {
                tr_str.to_string()
            }
        });

        // 若整张表格的所有行均为空行，直接清除该空表格
        let sanitized_table = TABLE_RE.replace_all(&no_empty_trs, |caps: &regex::Captures| {
            let t_str = &caps[0];
            let stripped = TAG_RE.replace_all(t_str, "");
            let clean = stripped.replace("&nbsp;", " ").replace("&#160;", " ").trim().to_string();
            if clean.is_empty() {
                String::new()
            } else {
                t_str.to_string()
            }
        }).to_string();

        let gfm = try_convert_to_gfm(&sanitized_table).unwrap_or_default();
        (gfm, sanitized_table)
    }

    #[cfg(feature = "table")]
    /// 使用二维面积重叠率 (IoF) 与空间阅读顺序拓扑保序将文本框分配给各个单元格
    pub fn assign_texts_to_cells(
        cell_boxes: &[crate::models::table::CellBox],
        texts: &[TextBoxItem],
    ) -> HashMap<usize, String> {
        if cell_boxes.is_empty() || texts.is_empty() {
            return HashMap::new();
        }

        let mut cell_items: HashMap<usize, Vec<&TextBoxItem>> = HashMap::new();

        for item in texts {
            let ix1 = item.coords[0];
            let iy1 = item.coords[1];
            let ix2 = item.coords[2];
            let iy2 = item.coords[3];
            let item_w = (ix2 - ix1).max(0.0);
            let item_h = (iy2 - iy1).max(0.0);
            let item_area = (item_w * item_h).max(1.0);

            let mut best_cell: Option<usize> = None;
            let mut max_iof = 0.0f32;
            let mut min_edge_dist = f32::INFINITY;
            let mut closest_cell: Option<usize> = None;

            for cell in cell_boxes {
                // 1. 计算文本框与单元格的二维交集面积
                let inter_x1 = ix1.max(cell.x1);
                let inter_y1 = iy1.max(cell.y1);
                let inter_x2 = ix2.min(cell.x2);
                let inter_y2 = iy2.min(cell.y2);

                let inter_w = (inter_x2 - inter_x1).max(0.0);
                let inter_h = (inter_y2 - inter_y1).max(0.0);
                let inter_area = inter_w * inter_h;

                let iof = inter_area / item_area;
                if iof > max_iof {
                    max_iof = iof;
                    best_cell = Some(cell.cell_idx);
                }

                // 2. 备用：计算中心点到单元格矩形边界的最短几何距离
                let cx = (ix1 + ix2) / 2.0;
                let cy = (iy1 + iy2) / 2.0;
                let dx = if cx < cell.x1 { cell.x1 - cx } else if cx > cell.x2 { cx - cell.x2 } else { 0.0 };
                let dy = if cy < cell.y1 { cell.y1 - cy } else if cy > cell.y2 { cy - cell.y2 } else { 0.0 };
                let dist = dx.hypot(dy);
                if dist < min_edge_dist {
                    min_edge_dist = dist;
                    closest_cell = Some(cell.cell_idx);
                }
            }

            // 归属决断：若最大重叠率 >= 0.20，判定为该单元格；否则若点到矩形边缘距离小于行高 1.5 倍，归入最近单元格
            let target_cell = if max_iof >= 0.20 {
                best_cell
            } else if min_edge_dist <= item_h * 1.5 {
                closest_cell.or(best_cell)
            } else {
                best_cell.or(closest_cell)
            };

            if let Some(c_idx) = target_cell {
                cell_items.entry(c_idx).or_default().push(item);
            }
        }

        // 对每个单元格内的文本框按空间阅读顺序几何排序 (先上后下，同行先左后右)
        let mut aggregated: HashMap<usize, String> = HashMap::new();
        for (c_idx, mut list) in cell_items {
            list.sort_by(|a, b| {
                let a_cy = (a.coords[1] + a.coords[3]) / 2.0;
                let b_cy = (b.coords[1] + b.coords[3]) / 2.0;
                let h = (a.coords[3] - a.coords[1]).min(b.coords[3] - b.coords[1]).max(5.0);
                if (a_cy - b_cy).abs() < h * 0.45 {
                    a.coords[0].partial_cmp(&b.coords[0]).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a_cy.partial_cmp(&b_cy).unwrap_or(std::cmp::Ordering::Equal)
                }
            });

            let text_parts: Vec<&str> = list.iter().map(|it| it.text.trim()).filter(|s| !s.is_empty()).collect();
            aggregated.insert(c_idx, text_parts.join(" "));
        }

        aggregated
    }
}

#[cfg(feature = "table")]
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// 将标准或带合并属性的 HTML `<table>` 转换为整齐对齐的 GFM 管道符表格
#[cfg(any(feature = "table", test))]
pub fn try_convert_to_gfm(html: &str) -> Option<String> {
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
            let tag_name = tag_lower.split_whitespace().next().unwrap_or("");
            if tag_name == "tr" {
                current_row = Vec::new();
            } else if tag_name == "/tr" {
                if !current_row.is_empty() && current_row.iter().any(|s| !s.trim().is_empty()) {
                    rows.push(current_row.clone());
                }
            } else if tag_name == "td" || tag_name == "th" || tag_name.starts_with("td") || tag_name.starts_with("th") {
                in_td = true;
                current_cell = String::new();
            } else if tag_name == "/td" || tag_name == "/th" {
                in_td = false;
                current_row.push(current_cell.trim().replace('\n', " ").replace('|', "\\|"));
            } else if in_td && (tag_name == "br" || tag_name.starts_with("br/")) {
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

    #[test]
    fn test_html_with_colspan_to_gfm() {
        let html = "<table><tr><td colspan=\"2\">合并表头</td></tr><tr><td>A</td><td>B</td></tr></table>";
        let gfm = try_convert_to_gfm(html).unwrap();
        assert!(gfm.contains("| 合并表头 |"));
        assert!(gfm.contains("| A | B |"));
    }
}
