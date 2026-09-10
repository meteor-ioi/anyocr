use crate::types::DocBlock;

/// Markdown 结构化序列化器
pub struct MarkdownSerializer;

impl MarkdownSerializer {
    /// 将 AST 语法块序列转换为格式整洁的标准 GFM Markdown 字符串
    pub fn serialize(blocks: &[DocBlock]) -> String {
        let mut sections: Vec<String> = Vec::new();

        for block in blocks {
            let md = block.to_markdown();
            let trimmed = md.trim();
            if !trimmed.is_empty() {
                sections.push(trimmed.to_string());
            }
        }

        // 块与块之间保留单个空行分隔
        let joined = sections.join("\n\n");
        let mut clean = String::with_capacity(joined.len());

        // 消除 3 个以上连续的换行符
        let mut newline_count = 0;
        for c in joined.chars() {
            if c == '\n' {
                newline_count += 1;
                if newline_count <= 2 {
                    clean.push(c);
                }
            } else {
                newline_count = 0;
                clean.push(c);
            }
        }

        clean.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_blocks() {
        let blocks = vec![
            DocBlock::Heading {
                level: 1,
                text: "文档主标题".into(),
                bbox: [0.0, 0.0, 100.0, 30.0],
            },
            DocBlock::Paragraph {
                lines: vec![],
                bbox: [0.0, 40.0, 100.0, 80.0],
            },
            DocBlock::Table {
                markdown_table: "| A | B |\n| --- | --- |\n| 1 | 2 |".into(),
                raw_html: None,
                bbox: [0.0, 90.0, 100.0, 150.0],
            },
        ];

        let md = MarkdownSerializer::serialize(&blocks);
        assert!(md.starts_with("# 文档主标题"));
        assert!(md.contains("| A | B |"));
    }
}
