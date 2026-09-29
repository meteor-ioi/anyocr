pub mod ast;
pub mod grid_table;
pub mod heading;
pub mod md_serializer;
pub mod para_merger;
pub mod reading_order;
pub mod table_matcher;

pub use ast::{compute_items_bbox, union_bbox};
pub use grid_table::GridTableBuilder;
pub use heading::HeadingClassifier;
pub use md_serializer::MarkdownSerializer;
pub use para_merger::ParagraphMerger;
pub use reading_order::ReadingOrder;
pub use table_matcher::TableMatcher;

use crate::types::{DocBlock, TextBoxItem};
#[cfg(feature = "table")]
use crate::models::table::TableStructureResult;
use std::collections::HashSet;

/// 版面重构与 AST 生成引擎
pub struct LayoutEngine;

impl LayoutEngine {
    /// 基于版面检测 (PicoDet-S) 与局部表格 ROI 执行高保真双轨版面还原
    pub fn process_dual_track(
        items: &[TextBoxItem],
        page_dims: (u32, u32),
        _layout_boxes: &[crate::models::LayoutBox],
        #[cfg(feature = "table")]
        roi_tables: &[(crate::models::LayoutBox, TableStructureResult)],
    ) -> (Vec<DocBlock>, String) {
        if items.is_empty() {
            return (Vec::new(), String::new());
        }

        let sorted_items = ReadingOrder::sort_reading_order(items, page_dims.0);
        let median_h = HeadingClassifier::compute_body_median_height(items);

        #[cfg(feature = "table")]
        if !roi_tables.is_empty() {
            // 1. 构建每个局部表格 Block
            let mut used_item_indices = HashSet::new();
            let mut table_blocks_with_y: Vec<(f32, DocBlock)> = Vec::new();

            for (lb, t_res) in roi_tables {
                let mut in_table_items = Vec::new();
                for (idx, item) in sorted_items.iter().enumerate() {
                    let cx = (item.coords[0] + item.coords[2]) / 2.0;
                    let cy = (item.coords[1] + item.coords[3]) / 2.0;

                    // 若中心点落在 Table ROI 内部 (略微外扩 5 像素缓冲)
                    if cx >= lb.x1 - 5.0 && cx <= lb.x2 + 5.0 && cy >= lb.y1 - 5.0 && cy <= lb.y2 + 5.0 {
                        in_table_items.push(item.clone());
                        used_item_indices.insert(idx);
                    }
                }

                if let Some(t_block) = TableMatcher::build_table_from_roi(t_res, &in_table_items, lb.to_array()) {
                    table_blocks_with_y.push((lb.y1, t_block));
                }
            }

            // 2. 将非表格文本行与各个 Table Block 按照 Y 轴阅读顺序流式交织组装
            table_blocks_with_y.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut final_blocks: Vec<DocBlock> = Vec::new();
            let mut current_text_batch: Vec<TextBoxItem> = Vec::new();
            let mut table_iter = table_blocks_with_y.into_iter().peekable();

            for (idx, item) in sorted_items.iter().enumerate() {
                if used_item_indices.contains(&idx) {
                    continue;
                }

                let item_y = item.coords[1];

                // 检查是否有表格的起始 Y 位于当前文本行之前
                while let Some((tbl_y, _)) = table_iter.peek() {
                    if *tbl_y <= item_y {
                        if !current_text_batch.is_empty() {
                            Self::process_text_region(&current_text_batch, median_h, &mut final_blocks);
                            current_text_batch.clear();
                        }
                        let (_, tbl_block) = table_iter.next().unwrap();
                        final_blocks.push(tbl_block);
                    } else {
                        break;
                    }
                }

                current_text_batch.push(item.clone());
            }

            // 结算剩余文本与剩余表格
            if !current_text_batch.is_empty() {
                Self::process_text_region(&current_text_batch, median_h, &mut final_blocks);
            }
            while let Some((_, tbl_block)) = table_iter.next() {
                final_blocks.push(tbl_block);
            }

            let markdown = MarkdownSerializer::serialize(&final_blocks);
            return (final_blocks, markdown);
        }

        // 降级回单轨/常规流程
        #[cfg(feature = "table")]
        {
            Self::process(items, page_dims, None)
        }
        #[cfg(not(feature = "table"))]
        {
            Self::process(items, page_dims)
        }
    }

    /// 执行端到端版面还原：
    /// 阅读顺序拓扑排序 ➔ 智能表格划分(SLANet/Grid融合) ➔ 标题定级 ➔ 段落/键值对规整 ➔ 组装 AST 与标准 Markdown
    pub fn process(
        items: &[TextBoxItem],
        page_dims: (u32, u32),
        #[cfg(feature = "table")]
        table_res: Option<&TableStructureResult>,
    ) -> (Vec<DocBlock>, String) {
        if items.is_empty() {
            return (Vec::new(), String::new());
        }

        // 1. 阅读顺序重排 (检测双栏或自上而下排序)
        let sorted_items = ReadingOrder::sort_reading_order(items, page_dims.0);
        let median_h = HeadingClassifier::compute_body_median_height(items);

        // 2. 尝试使用 SLANet 预测结果进行表格构建 (需通过合法性熔断校验)
        #[cfg(feature = "table")]
        let slanet_outcome = if let Some(res) = table_res {
            if TableMatcher::is_slanet_valid(res, &sorted_items) {
                Some(TableMatcher::match_and_build(res, &sorted_items))
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(not(feature = "table"))]
        let slanet_outcome: Option<(Vec<TextBoxItem>, Option<DocBlock>, Vec<TextBoxItem>)> = None;

        let mut final_blocks: Vec<DocBlock> = Vec::new();

        if let Some((header_items, table_block, footer_items)) = slanet_outcome {
            // A. SLANet 结构有效，走标准三段式还原
            Self::process_text_region(&header_items, median_h, &mut final_blocks);
            if let Some(table) = table_block {
                final_blocks.push(table);
            }
            Self::process_text_region(&footer_items, median_h, &mut final_blocks);
        } else {
            // B. SLANet 无效/欠分割/未启用时，启动启发式坐标网格多表格提取器 (GridTableBuilder)
            let detected_tables = GridTableBuilder::extract_grid_tables(&sorted_items, page_dims);

            if detected_tables.is_empty() {
                // 无表格，全图作为正文/键值对流解析
                Self::process_text_region(&sorted_items, median_h, &mut final_blocks);
            } else {
                // 存在 1 个或多个表格：流式交织组装 AST
                Self::process_multi_table_flow(&sorted_items, &detected_tables, median_h, &mut final_blocks);
            }
        }

        // 3. 序列化为规范标准 Markdown
        let markdown = MarkdownSerializer::serialize(&final_blocks);

        (final_blocks, markdown)
    }

    /// 解析一段顺序文本行为标题、自然段落与列表
    fn process_text_region(
        items: &[TextBoxItem],
        median_h: f32,
        out_blocks: &mut Vec<DocBlock>,
    ) {
        let mut non_heading_batch: Vec<TextBoxItem> = Vec::new();

        for item in items {
            if let Some(level) = HeadingClassifier::classify_heading(item, median_h) {
                // 遇到标题时，先结算之前的非标题行批次
                if !non_heading_batch.is_empty() {
                    let para_blocks = ParagraphMerger::merge_lines_into_blocks(&non_heading_batch, median_h);
                    out_blocks.extend(para_blocks);
                    non_heading_batch.clear();
                }

                out_blocks.push(DocBlock::Heading {
                    level,
                    text: item.text.trim().to_string(),
                    bbox: item.coords,
                });
            } else {
                non_heading_batch.push(item.clone());
            }
        }

        if !non_heading_batch.is_empty() {
            let para_blocks = ParagraphMerger::merge_lines_into_blocks(&non_heading_batch, median_h);
            out_blocks.extend(para_blocks);
        }
    }

    /// 多表格与正文流式交织组装
    fn process_multi_table_flow(
        all_items: &[TextBoxItem],
        detected_tables: &[grid_table::DetectedTable],
        median_h: f32,
        out_blocks: &mut Vec<DocBlock>,
    ) {
        let mut used_indices: HashSet<usize> = HashSet::new();
        for t in detected_tables {
            for &idx in &t.used_item_indices {
                used_indices.insert(idx);
            }
        }

        // 提取未被表格占用的游离文本框 (保序)
        let free_items: Vec<TextBoxItem> = all_items
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_indices.contains(i))
            .map(|(_, it)| it.clone())
            .collect();

        // 收集所有表格节点及其垂直区间
        let mut sorted_tables = detected_tables.to_vec();
        sorted_tables.sort_by(|a, b| a.start_y.total_cmp(&b.start_y));

        let mut current_free_idx = 0;
        let num_free = free_items.len();

        for table in &sorted_tables {
            // 将位于当前表格上方的正文先输出
            let mut region_before = Vec::new();
            while current_free_idx < num_free {
                let item = &free_items[current_free_idx];
                let cy = (item.coords[1] + item.coords[3]) / 2.0;
                if cy < table.start_y - 2.0 {
                    region_before.push(item.clone());
                    current_free_idx += 1;
                } else {
                    break;
                }
            }

            if !region_before.is_empty() {
                Self::process_text_region(&region_before, median_h, out_blocks);
            }

            // 输出当前表格 AST 节点
            out_blocks.push(DocBlock::Table {
                markdown_table: table.markdown.clone(),
                raw_html: Some(table.html.clone()),
                bbox: table.bbox,
            });
        }

        // 将位于最后一个表格下方的剩余正文输出
        if current_free_idx < num_free {
            let remaining = &free_items[current_free_idx..];
            Self::process_text_region(remaining, median_h, out_blocks);
        }
    }
}
