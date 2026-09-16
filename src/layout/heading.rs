use crate::types::TextBoxItem;

/// 标题层级识别与分类器
pub struct HeadingClassifier;

impl HeadingClassifier {
    /// 计算单页所有文本框高度的中位数，作为正文基准字号 $H_{body}$
    pub fn compute_body_median_height(items: &[TextBoxItem]) -> f32 {
        if items.is_empty() {
            return 24.0;
        }

        let mut heights: Vec<f32> = items
            .iter()
            .map(|item| (item.coords[3] - item.coords[1]).abs())
            .filter(|&h| h > 4.0)
            .collect();

        if heights.is_empty() {
            return 24.0;
        }

        heights.sort_by(|a, b| a.total_cmp(b));
        heights[heights.len() / 2]
    }

    /// 判定一个文本框是否是标题，若是则返回对应 Markdown 级别 (1 ~ 6)
    pub fn classify_heading(item: &TextBoxItem, body_median_height: f32) -> Option<u8> {
        let text = item.text.trim();
        if text.is_empty() || text.starts_with('-') || text.starts_with('+') || text.starts_with('*') || text.starts_with('~') || text.starts_with('=') {
            return None;
        }

        let h = (item.coords[3] - item.coords[1]).abs();
        let char_count = text.chars().count();

        // 排除过长文本（超过 50 个字极少作为一级或二级标题）
        if char_count > 60 {
            return None;
        }

        // 1. 基于相对行高的聚类判定
        if h >= 1.8 * body_median_height && char_count <= 40 {
            return Some(1);
        }
        if h >= 1.4 * body_median_height && char_count <= 50 {
            return Some(2);
        }

        // 2. 基于行首编号与语义规则的标题提权
        if Self::matches_section_heading_pattern(text) {
            if h >= 1.15 * body_median_height {
                return Some(2);
            } else {
                return Some(3);
            }
        }

        // 3. 特殊单据主标题特征识别（如居中较短包含"表"、"单"、"书"、"通知"）
        if char_count <= 25
            && (text.ends_with("单")
                || text.ends_with("表")
                || text.ends_with("发票")
                || text.ends_with("通知")
                || text.ends_with("规定")
                || text.ends_with("合同"))
            && h >= 1.25 * body_median_height
        {
            return Some(1);
        }

        None
    }

    /// 匹配章节或编号式标题规则 (如 "第一章", "一、", "1.1", "（一）")
    fn matches_section_heading_pattern(text: &str) -> bool {
        // "第一章", "第一节", "第一条"
        if text.starts_with("第")
            && (text.contains('章') || text.contains('节') || text.contains('条') || text.contains('篇'))
            && text.chars().count() <= 35
        {
            return true;
        }

        // "一、", "二、", "三、" ...
        let cn_nums = ["一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、"];
        for prefix in &cn_nums {
            if text.starts_with(prefix) {
                return true;
            }
        }

        // "(一)", "（一）"
        if (text.starts_with("（一）") || text.starts_with("(一)") || text.starts_with("（1）") || text.starts_with("(1)"))
            && text.chars().count() <= 35
        {
            return true;
        }

        // "1. ", "1.1 ", "1.1.1 " 且后面紧跟文字
        if let Some(first_char) = text.chars().next() {
            if first_char.is_ascii_digit() && (text.contains(". ") || text.contains("、")) {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_height() {
        let items = vec![
            TextBoxItem { text: "正文1".into(), score: 0.9, coords: [0.0, 10.0, 100.0, 30.0] }, // h = 20
            TextBoxItem { text: "正文2".into(), score: 0.9, coords: [0.0, 35.0, 100.0, 55.0] }, // h = 20
            TextBoxItem { text: "标题".into(), score: 0.9, coords: [0.0, 0.0, 100.0, 40.0] },   // h = 40
        ];
        let median = HeadingClassifier::compute_body_median_height(&items);
        assert_eq!(median, 20.0);
    }

    #[test]
    fn test_heading_classification() {
        let title_item = TextBoxItem {
            text: "采购申请单".into(),
            score: 0.95,
            coords: [50.0, 10.0, 200.0, 50.0], // h = 40 (2x of 20)
        };
        let level = HeadingClassifier::classify_heading(&title_item, 20.0);
        assert_eq!(level, Some(1));

        let chapter_item = TextBoxItem {
            text: "第一章 采购审批权限".into(),
            score: 0.95,
            coords: [50.0, 60.0, 250.0, 85.0], // h = 25 (1.25x of 20)
        };
        let level = HeadingClassifier::classify_heading(&chapter_item, 20.0);
        assert_eq!(level, Some(2));
    }
}
