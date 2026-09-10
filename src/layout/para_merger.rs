use crate::layout::ast::compute_items_bbox;
use crate::types::{DocBlock, ListItem, TextBoxItem};

/// 段落截断标点集 (末尾包含这些标点时表示自然句末，不进行跨行合并)
pub const LINE_STOP_FLAGS: &[char] = &[
    '.', '!', '?', '。', '！', '？', ':', '：', ';', '；',
];

/// 段落合并器与自然折行重建
pub struct ParagraphMerger;

impl ParagraphMerger {
    /// 判定一个文本行末尾是否包含截断标点
    pub fn ends_with_stop_flag(text: &str) -> bool {
        let trimmed = text.trim_end();
        if let Some(last_char) = trimmed.chars().last() {
            LINE_STOP_FLAGS.contains(&last_char)
        } else {
            true
        }
    }

    /// 智能拼接两行文本（中文字符无缝连接，英文字符补充单个空格）
    pub fn join_text_smart(first: &str, second: &str) -> String {
        let t1 = first.trim_end();
        let t2 = second.trim_start();

        if t1.is_empty() {
            return t2.to_string();
        }
        if t2.is_empty() {
            return t1.to_string();
        }

        // 处理英文连字符 (如 "imple-\nmentation" ➔ "implementation")
        if t1.ends_with('-') {
            let without_hyphen = &t1[..t1.len() - 1];
            if without_hyphen.chars().last().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
                && t2.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
            {
                return format!("{}{}", without_hyphen, t2);
            }
        }

        let last_c = t1.chars().last().unwrap();
        let first_c = t2.chars().next().unwrap();

        // 中文/全角文字连接无需添加空格
        let is_cjk1 = is_cjk_char(last_c);
        let is_cjk2 = is_cjk_char(first_c);

        if is_cjk1 && is_cjk2 {
            format!("{}{}", t1, t2)
        } else {
            format!("{} {}", t1, t2)
        }
    }

    /// 将连续的文本行聚类合并为段落或列表块
    pub fn merge_lines_into_blocks(items: &[TextBoxItem], median_height: f32) -> Vec<DocBlock> {
        if items.is_empty() {
            return Vec::new();
        }

        let mut blocks = Vec::new();
        let mut current_para: Vec<TextBoxItem> = Vec::new();

        for item in items {
            let text = item.text.trim();
            if text.is_empty() {
                continue;
            }

            // 1. 检查是否为列表项 (如 "- ", "* ", "1. ", "• ")
            if let Some((ordered, content)) = parse_list_prefix(text) {
                // 先结算当前段落
                if !current_para.is_empty() {
                    let bbox = compute_items_bbox(&current_para);
                    blocks.push(DocBlock::Paragraph {
                        lines: std::mem::take(&mut current_para),
                        bbox,
                    });
                }

                // 创建或追加到最后一个 List 块
                let list_item = ListItem {
                    text: content.to_string(),
                    bbox: item.coords,
                };

                let mut appended = false;
                if let Some(DocBlock::List { ordered: last_ordered, items: list_items, bbox }) = blocks.last_mut() {
                    if *last_ordered == ordered {
                        list_items.push(list_item.clone());
                        *bbox = crate::layout::ast::union_bbox(bbox, &item.coords);
                        appended = true;
                    }
                }

                if !appended {
                    blocks.push(DocBlock::List {
                        ordered,
                        items: vec![list_item],
                        bbox: item.coords,
                    });
                }
                continue;
            }

            // 2. 判定是否与当前段落合并
            if let Some(last_item) = current_para.last() {
                let vertical_dist = (item.coords[1] - last_item.coords[3]).max(0.0);
                let is_stopped = Self::ends_with_stop_flag(&last_item.text);

                // 启发式跨行条件：
                // a) 上一行未结束断句
                // b) 行间距不超过基准行高的 1.6 倍
                if !is_stopped && vertical_dist <= 1.6 * median_height {
                    current_para.push(item.clone());
                } else {
                    // 结算上一段落并开启新段落
                    let bbox = compute_items_bbox(&current_para);
                    blocks.push(DocBlock::Paragraph {
                        lines: std::mem::take(&mut current_para),
                        bbox,
                    });
                    current_para.push(item.clone());
                }
            } else {
                current_para.push(item.clone());
            }
        }

        // 结算最后剩余的段落
        if !current_para.is_empty() {
            let bbox = compute_items_bbox(&current_para);
            blocks.push(DocBlock::Paragraph {
                lines: current_para,
                bbox,
            });
        }

        blocks
    }
}

/// 判定是否为 CJK 汉字或全角标点
fn is_cjk_char(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |
        '\u{3400}'..='\u{4DBF}' |
        '\u{3000}'..='\u{303F}' |
        '\u{FF00}'..='\u{FFEF}'
    )
}

/// 解析文本前缀是否为列表项
fn parse_list_prefix(text: &str) -> Option<(bool, &str)> {
    if let Some(rest) = text.strip_prefix("- ").or_else(|| text.strip_prefix("* ")).or_else(|| text.strip_prefix("• ")) {
        return Some((false, rest.trim()));
    }

    // "1. ", "2. " 有序列表检测
    let mut chars = text.chars().peekable();
    let mut num_len = 0;
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            num_len += 1;
            chars.next();
        } else {
            break;
        }
    }

    if num_len > 0 && num_len <= 3 {
        let rest = &text[num_len..];
        if let Some(content) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix("、")) {
            return Some((true, content.trim()));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_text_smart() {
        assert_eq!(
            ParagraphMerger::join_text_smart("深度学习技术", "正在快速变革"),
            "深度学习技术正在快速变革"
        );
        assert_eq!(
            ParagraphMerger::join_text_smart("High performance", "OCR engine"),
            "High performance OCR engine"
        );
        assert_eq!(
            ParagraphMerger::join_text_smart("trans-", "former"),
            "transformer"
        );
    }

    #[test]
    fn test_paragraph_merge() {
        let lines = vec![
            TextBoxItem {
                text: "这是一个很长的句子正在跨行折".into(),
                score: 0.9,
                coords: [10.0, 10.0, 300.0, 30.0],
            },
            TextBoxItem {
                text: "行显示，本应该属于同一个段落。".into(),
                score: 0.9,
                coords: [10.0, 35.0, 300.0, 55.0],
            },
            TextBoxItem {
                text: "这是全新的第二个段落。".into(),
                score: 0.9,
                coords: [10.0, 80.0, 200.0, 100.0],
            },
        ];

        let blocks = ParagraphMerger::merge_lines_into_blocks(&lines, 20.0);
        assert_eq!(blocks.len(), 2);
        if let DocBlock::Paragraph { lines, .. } = &blocks[0] {
            assert_eq!(lines.len(), 2);
        } else {
            panic!("应为段落块");
        }
    }
}
