use crate::types::{DocBlock, TextBoxItem};

/// 计算两个包围盒的最小外接矩形 (Union Bounding Box)
pub fn union_bbox(b1: &[f32; 4], b2: &[f32; 4]) -> [f32; 4] {
    [
        b1[0].min(b2[0]),
        b1[1].min(b2[1]),
        b1[2].max(b2[2]),
        b1[3].max(b2[3]),
    ]
}

/// 计算一组文本框的整体最小外接矩形
pub fn compute_items_bbox(items: &[TextBoxItem]) -> [f32; 4] {
    if items.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let mut b = items[0].coords;
    for item in &items[1..] {
        b = union_bbox(&b, &item.coords);
    }
    b
}

impl DocBlock {
    /// 获取当前 AST 节点的空间几何包围盒
    pub fn bbox(&self) -> [f32; 4] {
        match self {
            DocBlock::Heading { bbox, .. } => *bbox,
            DocBlock::Paragraph { bbox, .. } => *bbox,
            DocBlock::Table { bbox, .. } => *bbox,
            DocBlock::List { bbox, .. } => *bbox,
            DocBlock::Image { bbox, .. } => *bbox,
        }
    }

    /// 将单个 AST 节点序列化为 Markdown 字符串片段
    pub fn to_markdown(&self) -> String {
        match self {
            DocBlock::Heading { level, text, .. } => {
                let hashes = "#".repeat((*level as usize).clamp(1, 6));
                format!("{} {}", hashes, text.trim())
            }
            DocBlock::Paragraph { lines, .. } => {
                let joined_text = lines
                    .iter()
                    .map(|l| l.text.trim())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                joined_text
            }
            DocBlock::Table { markdown_table, raw_html, .. } => {
                if !markdown_table.trim().is_empty() {
                    markdown_table.trim().to_string()
                } else if let Some(html) = raw_html {
                    html.trim().to_string()
                } else {
                    String::new()
                }
            }
            DocBlock::List { ordered, items, .. } => {
                let mut out = String::new();
                for (idx, item) in items.iter().enumerate() {
                    let prefix = if *ordered {
                        format!("{}. ", idx + 1)
                    } else {
                        "- ".to_string()
                    };
                    out.push_str(&format!("{}{}\n", prefix, item.text.trim()));
                }
                out.trim_end().to_string()
            }
            DocBlock::Image { format: _, alt, .. } => {
                let alt_text = alt.as_deref().unwrap_or("image");
                format!("![{}]()", alt_text)
            }
        }
    }
}
