pub mod ast;
pub mod heading;
pub mod md_serializer;
pub mod para_merger;
pub mod reading_order;
pub mod table_matcher;

pub use ast::{compute_items_bbox, union_bbox};
pub use heading::HeadingClassifier;
pub use md_serializer::MarkdownSerializer;
pub use para_merger::ParagraphMerger;
pub use reading_order::ReadingOrder;
pub use table_matcher::TableMatcher;

use crate::types::{DocBlock, TextBoxItem};
#[cfg(feature = "table")]
use crate::models::table::TableStructureResult;

/// 版面重构与 AST 生成引擎
pub struct LayoutEngine;

impl LayoutEngine {
    /// 执行端到端版面还原：
    /// 阅读顺序拓扑排序 ➔ 表格划分 ➔ 标题自适应定级 ➔ 自然段落折行合并 ➔ 组装 AST 与标准 Markdown
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

        // 2. 表格与正文区域划分
        #[cfg(feature = "table")]
        let (header_items, table_block, footer_items) = if let Some(res) = table_res {
            TableMatcher::match_and_build(res, &sorted_items)
        } else {
            (sorted_items, None, Vec::new())
        };

        #[cfg(not(feature = "table"))]
        let (header_items, table_block, footer_items) = (sorted_items, None::<DocBlock>, Vec::<TextBoxItem>::new());

        // 3. 计算页面正文基准行高中位数
        let median_h = HeadingClassifier::compute_body_median_height(items);

        let mut final_blocks: Vec<DocBlock> = Vec::new();

        // 4. 解析表格前正文区 (Header)
        Self::process_text_region(&header_items, median_h, &mut final_blocks);

        // 5. 插入结构化表格块
        if let Some(table) = table_block {
            final_blocks.push(table);
        }

        // 6. 解析表格后正文区 (Footer)
        Self::process_text_region(&footer_items, median_h, &mut final_blocks);

        // 7. 序列化为规范标准 Markdown
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
}
