use crate::types::TextBoxItem;

/// 启发式几何网格表格构建器 (基于 OCR 文本框坐标投影与列对齐聚类)
pub struct GridTableBuilder;

/// 表格区域检测产物
#[derive(Debug, Clone)]
pub struct DetectedTable {
    pub start_y: f32,
    pub end_y: f32,
    pub bbox: [f32; 4],
    pub markdown: String,
    pub html: String,
    pub used_item_indices: Vec<usize>,
}

/// 逻辑文本行 (按 Y 轴重叠聚类)
#[derive(Debug, Clone)]
struct LogicalRow {
    min_y: f32,
    max_y: f32,
    items: Vec<(usize, TextBoxItem)>, // (原始索引, item)
}

impl GridTableBuilder {
    /// 从页面文本框集合中自动检测并抽取所有结构化网格表格
    pub fn extract_grid_tables(
        items: &[TextBoxItem],
        _page_dims: (u32, u32),
    ) -> Vec<DetectedTable> {
        if items.len() < 4 {
            return Vec::new();
        }

        // 1. 将所有文本框附加原索引并按 Y 轴升序排序
        let mut indexed_items: Vec<(usize, TextBoxItem)> = items
            .iter()
            .enumerate()
            .map(|(i, it)| (i, it.clone()))
            .collect();
        indexed_items.sort_by(|a, b| a.1.coords[1].total_cmp(&b.1.coords[1]));

        // 2. 聚类为逻辑行
        let rows = Self::cluster_into_rows(&indexed_items);
        if rows.len() < 2 {
            return Vec::new();
        }

        // 3. 寻找具有多列网格特征的连续行区间 (Table Segments)
        let table_ranges = Self::find_table_row_ranges(&rows);

        let mut tables = Vec::new();
        for (start_r, end_r) in table_ranges {
            let row_slice = &rows[start_r..=end_r];
            if let Some(table) = Self::build_table_from_rows(row_slice) {
                tables.push(table);
            }
        }

        tables
    }

    /// 将文本框按垂直重叠聚类为逻辑行
    fn cluster_into_rows(indexed_items: &[(usize, TextBoxItem)]) -> Vec<LogicalRow> {
        let mut rows: Vec<LogicalRow> = Vec::new();

        for (orig_idx, item) in indexed_items {
            let y1 = item.coords[1];
            let y2 = item.coords[3];
            let item_h = (y2 - y1).max(1.0);

            let mut matched_row_idx = None;
            for (r_idx, row) in rows.iter().enumerate().rev().take(4) {
                let row_h = (row.max_y - row.min_y).max(1.0);
                let overlap = (y2.min(row.max_y) - y1.max(row.min_y)).max(0.0);
                let min_h = item_h.min(row_h);

                // 判定是否在同一行：垂直重叠超过 35% 或上下边缘极近 (< 6px)
                if overlap > 0.35 * min_h || (y1 - row.min_y).abs() < 6.0 {
                    matched_row_idx = Some(r_idx);
                    break;
                }
            }

            if let Some(r_idx) = matched_row_idx {
                let row = &mut rows[r_idx];
                row.min_y = row.min_y.min(y1);
                row.max_y = row.max_y.max(y2);
                row.items.push((*orig_idx, item.clone()));
            } else {
                rows.push(LogicalRow {
                    min_y: y1,
                    max_y: y2,
                    items: vec![(*orig_idx, item.clone())],
                });
            }
        }

        // 每行内按 X 轴坐标自左向右排序
        for row in &mut rows {
            row.items.sort_by(|a, b| a.1.coords[0].total_cmp(&b.1.coords[0]));
        }

        // 按 Y 轴升序整理各行
        rows.sort_by(|a, b| a.min_y.total_cmp(&b.min_y));
        rows
    }

    /// 判定行内是否为明显的非表格手写/注释/独立段落
    fn is_row_handwritten_or_note(row: &LogicalRow) -> bool {
        let text = row
            .items
            .iter()
            .map(|(_, it)| it.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let trimmed = text.trim();
        trimmed.starts_with("-Journal")
            || trimmed.starts_with("-cxc")
            || trimmed.starts_with("- rotal")
            || trimmed.starts_with("-Total")
            || trimmed.starts_with("Muon")
            || trimmed.starts_with("July")
    }

    /// 寻找满足表格特征的连续行区间 [start_row, end_row]
    fn find_table_row_ranges(rows: &[LogicalRow]) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let n = rows.len();
        let mut in_table = false;
        let mut start_idx = 0;

        for i in 0..n {
            let is_note = Self::is_row_handwritten_or_note(&rows[i]);
            let is_multi_col = !is_note && Self::is_row_multi_column(&rows[i]);
            let has_table_keyword = !is_note && Self::row_has_table_header_keyword(&rows[i]);

            if !in_table {
                // 表格开始条件：多列行且有关键词，或者连续 2 行以上多列
                if has_table_keyword || (is_multi_col && i + 1 < n && Self::is_row_multi_column(&rows[i + 1])) {
                    in_table = true;
                    start_idx = i;
                }
            } else {
                // 遇到手写笔记或非多列正文时表格结束
                if is_note || (!is_multi_col && !has_table_keyword) {
                    let next_is_multi = !is_note && i + 1 < n && Self::is_row_multi_column(&rows[i + 1]);
                    if !next_is_multi {
                        let row_count = i - start_idx;
                        if row_count >= 2 {
                            ranges.push((start_idx, i - 1));
                        }
                        in_table = false;
                    }
                }
            }
        }

        if in_table {
            let row_count = n - start_idx;
            if row_count >= 2 {
                ranges.push((start_idx, n - 1));
            }
        }

        ranges
    }

    /// 判定一行是否具有多列特征 (水平不重叠且间距明显)
    fn is_row_multi_column(row: &LogicalRow) -> bool {
        if row.items.len() >= 2 {
            let mut distinct_cols = 1;
            for w in row.items.windows(2) {
                let prev_x2 = w[0].1.coords[2];
                let curr_x1 = w[1].1.coords[0];
                if curr_x1 >= prev_x2 - 2.0 {
                    distinct_cols += 1;
                }
            }
            return distinct_cols >= 2;
        }
        false
    }

    /// 判定行内是否包含典型的表头关键词
    fn row_has_table_header_keyword(row: &LogicalRow) -> bool {
        let line_text = row
            .items
            .iter()
            .map(|(_, it)| it.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .to_uppercase();

        let keywords = [
            "DETAILED PARTICULARS", "INVOICE NO", "PO NO", "AMOUNT", "CCY",
            "DESCRIPTION", "QTY", "QUANTITY", "UNIT PRICE", "TOTAL AMOUNT",
            "细节描述", "发票号", "采购单号", "金额", "币种", "单价", "数量", "品名", "项目", "序号",
            "APPLIED BY", "CHECKED BY", "APPROVED BY", "VERIFIED BY", "RECEIVED BY",
            "申请人", "确认人", "核批人", "审核人", "签收人", "PAYMENT 应付金额"
        ];

        keywords.iter().any(|&k| line_text.contains(k))
    }

    /// 从一系列逻辑行构建对齐的表格 AST
    fn build_table_from_rows(rows: &[LogicalRow]) -> Option<DetectedTable> {
        if rows.is_empty() {
            return None;
        }

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut used_indices = Vec::new();

        for row in rows {
            for (idx, item) in &row.items {
                min_x = min_x.min(item.coords[0]);
                min_y = min_y.min(item.coords[1]);
                max_x = max_x.max(item.coords[2]);
                max_y = max_y.max(item.coords[3]);
                used_indices.push(*idx);
            }
        }

        // 1. 聚类出全局列区间 (Column Intervals)
        let col_intervals = Self::compute_column_intervals(rows);
        if col_intervals.is_empty() {
            return None;
        }

        let num_cols = col_intervals.len();

        // 2. 将每一行填充到对应的列槽位中
        let mut grid: Vec<Vec<String>> = Vec::new();
        for row in rows {
            let mut row_cells: Vec<Vec<String>> = vec![Vec::new(); num_cols];

            for (_, item) in &row.items {
                let cx = (item.coords[0] + item.coords[2]) / 2.0;

                // 寻找落入或中心距离最近的列区间
                let mut best_col = 0;
                let mut min_dist = f32::INFINITY;

                for (c_idx, (col_x1, col_x2)) in col_intervals.iter().enumerate() {
                    if cx >= *col_x1 && cx <= *col_x2 {
                        best_col = c_idx;
                        break;
                    }
                    let col_mid = (col_x1 + col_x2) / 2.0;
                    let dist = (cx - col_mid).abs();
                    if dist < min_dist {
                        min_dist = dist;
                        best_col = c_idx;
                    }
                }

                row_cells[best_col].push(item.text.trim().to_string());
            }

            let cell_strings: Vec<String> = row_cells
                .into_iter()
                .map(|texts| texts.join(" "))
                .collect();
            grid.push(cell_strings);
        }

        // 3. 序列化为 GFM Markdown 表格与标准 HTML
        let markdown = Self::render_grid_to_gfm(&grid);
        let html = Self::render_grid_to_html(&grid);

        Some(DetectedTable {
            start_y: min_y,
            end_y: max_y,
            bbox: [min_x, min_y, max_x, max_y],
            markdown,
            html,
            used_item_indices: used_indices,
        })
    }

    /// 基于各行内 items 的 X 坐标聚类出列区间 [col_start_x, col_end_x]
    fn compute_column_intervals(rows: &[LogicalRow]) -> Vec<(f32, f32)> {
        // 收集多列行中的文本框中心点 X 坐标
        let mut centers = Vec::new();
        for row in rows {
            if row.items.len() >= 2 {
                for (_, it) in &row.items {
                    let cx = (it.coords[0] + it.coords[2]) / 2.0;
                    centers.push((cx, it.coords[0], it.coords[2]));
                }
            }
        }

        if centers.is_empty() {
            for row in rows {
                for (_, it) in &row.items {
                    let cx = (it.coords[0] + it.coords[2]) / 2.0;
                    centers.push((cx, it.coords[0], it.coords[2]));
                }
            }
        }

        centers.sort_by(|a, b| a.0.total_cmp(&b.0));

        // 基于自适应距离进行列聚类 (中心相距 < 35px 归为同列)
        let mut clusters: Vec<Vec<(f32, f32, f32)>> = Vec::new();
        for c in centers {
            if let Some(last_cluster) = clusters.last_mut() {
                let cluster_cx = last_cluster.iter().map(|it| it.0).sum::<f32>() / last_cluster.len() as f32;
                if (c.0 - cluster_cx).abs() < 38.0 {
                    last_cluster.push(c);
                    continue;
                }
            }
            clusters.push(vec![c]);
        }

        // 过滤并保留出现频次 >= 2 或总行数较少时的列
        let mut valid_clusters = Vec::new();
        for cl in clusters {
            if cl.len() >= 2 || rows.len() <= 3 {
                let min_x = cl.iter().map(|it| it.1).fold(f32::INFINITY, f32::min);
                let max_x = cl.iter().map(|it| it.2).fold(f32::NEG_INFINITY, f32::max);
                valid_clusters.push((min_x - 5.0, max_x + 5.0));
            }
        }

        if valid_clusters.is_empty() {
            return Vec::new();
        }

        valid_clusters.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged_cols: Vec<(f32, f32)> = Vec::new();
        for col in valid_clusters {
            if let Some(last) = merged_cols.last_mut() {
                if col.0 <= last.1 {
                    last.1 = last.1.max(col.1);
                    continue;
                }
            }
            merged_cols.push(col);
        }

        merged_cols
    }

    /// 将 2D 字符串矩阵格式化为 GFM Markdown 管道表格
    fn render_grid_to_gfm(grid: &[Vec<String>]) -> String {
        if grid.is_empty() {
            return String::new();
        }

        let num_cols = grid.iter().map(|r| r.len()).max().unwrap_or(0);
        if num_cols == 0 {
            return String::new();
        }

        let mut out = String::new();
        for (r_idx, row) in grid.iter().enumerate() {
            let mut padded = row.clone();
            while padded.len() < num_cols {
                padded.push(String::new());
            }

            let escaped: Vec<String> = padded
                .iter()
                .map(|cell| cell.trim().replace('|', "\\|").replace('\n', " "))
                .collect();

            out.push_str(&format!("| {} |\n", escaped.join(" | ")));

            if r_idx == 0 {
                let sep = vec!["---"; num_cols];
                out.push_str(&format!("| {} |\n", sep.join(" | ")));
            }
        }

        out.trim_end().to_string()
    }

    /// 将 2D 字符串矩阵格式化为干净语义 HTML 表格
    fn render_grid_to_html(grid: &[Vec<String>]) -> String {
        if grid.is_empty() {
            return String::new();
        }

        let mut html = String::from("<table>");
        for (r_idx, row) in grid.iter().enumerate() {
            html.push_str("<tr>");
            let tag = if r_idx == 0 { "th" } else { "td" };
            for cell in row {
                let clean_text = cell.trim().replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
                html.push_str(&format!("<{}>{}</{}>", tag, clean_text, tag));
            }
            html.push_str("</tr>");
        }
        html.push_str("</table>");
        html
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_table_extraction() {
        let items = vec![
            TextBoxItem { text: "序号".into(), score: 0.9, coords: [100.0, 100.0, 150.0, 120.0] },
            TextBoxItem { text: "品名".into(), score: 0.9, coords: [200.0, 100.0, 300.0, 120.0] },
            TextBoxItem { text: "金额".into(), score: 0.9, coords: [400.0, 100.0, 500.0, 120.0] },

            TextBoxItem { text: "1".into(), score: 0.9, coords: [100.0, 130.0, 150.0, 150.0] },
            TextBoxItem { text: "办公用品".into(), score: 0.9, coords: [200.0, 130.0, 300.0, 150.0] },
            TextBoxItem { text: "$ 100".into(), score: 0.9, coords: [400.0, 130.0, 500.0, 150.0] },

            TextBoxItem { text: "2".into(), score: 0.9, coords: [100.0, 160.0, 150.0, 180.0] },
            TextBoxItem { text: "电子耗材".into(), score: 0.9, coords: [200.0, 160.0, 300.0, 180.0] },
            TextBoxItem { text: "$ 250".into(), score: 0.9, coords: [400.0, 160.0, 500.0, 180.0] },
        ];

        let tables = GridTableBuilder::extract_grid_tables(&items, (1000, 1000));
        assert_eq!(tables.len(), 1);
        assert!(tables[0].markdown.contains("| 序号 | 品名 | 金额 |"));
        assert!(tables[0].markdown.contains("| 1 | 办公用品 | $ 100 |"));
        assert!(tables[0].markdown.contains("| 2 | 电子耗材 | $ 250 |"));
    }
}

