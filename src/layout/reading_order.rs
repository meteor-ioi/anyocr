use crate::types::TextBoxItem;

/// 阅读顺序重构与多栏分流排序器
pub struct ReadingOrder;

impl ReadingOrder {
    /// 对单页文本行进行阅读顺序拓扑排序 (支持双栏文档检测与分流)
    pub fn sort_reading_order(items: &[TextBoxItem], page_width: u32) -> Vec<TextBoxItem> {
        if items.len() <= 3 || page_width == 0 {
            let mut sorted = items.to_vec();
            sorted.sort_by(|a, b| a.coords[1].total_cmp(&b.coords[1]));
            return sorted;
        }

        // 1. 尝试检测是否存在垂直双栏中缝 (White-space Valley)
        if let Some((gap_x1, gap_x2)) = Self::detect_vertical_valley(items, page_width) {
            let mut top_span = Vec::new();
            let mut left_col = Vec::new();
            let mut right_col = Vec::new();
            let mut bottom_span = Vec::new();

            // 计算左右分栏的中线
            let mid_x = (gap_x1 + gap_x2) / 2.0;

            for item in items {
                let x1 = item.coords[0];
                let x2 = item.coords[2];
                let width = (x2 - x1).max(0.0);

                // 若文本宽度跨越了整个中缝或者跨越了半个页面以上，判定为通栏
                if (x1 < gap_x1 && x2 > gap_x2) || width > page_width as f32 * 0.65 {
                    if item.coords[1] < (page_width as f32) * 0.4 {
                        top_span.push(item.clone());
                    } else {
                        bottom_span.push(item.clone());
                    }
                } else if x2 <= mid_x + 5.0 {
                    left_col.push(item.clone());
                } else {
                    right_col.push(item.clone());
                }
            }

            // 各栏内部按垂直 y1 升序排序
            top_span.sort_by(|a, b| a.coords[1].total_cmp(&b.coords[1]));
            left_col.sort_by(|a, b| a.coords[1].total_cmp(&b.coords[1]));
            right_col.sort_by(|a, b| a.coords[1].total_cmp(&b.coords[1]));
            bottom_span.sort_by(|a, b| a.coords[1].total_cmp(&b.coords[1]));

            // 拓扑组装：顶通栏 ➔ 左栏 ➔ 右栏 ➔ 底通栏
            let mut result = Vec::with_capacity(items.len());
            result.extend(top_span);
            result.extend(left_col);
            result.extend(right_col);
            result.extend(bottom_span);
            return result;
        }

        // 无分栏特征时，采用自上而下自左向右自然排序
        let mut sorted = items.to_vec();
        sorted.sort_by(|a, b| {
            let y_diff = (a.coords[1] - b.coords[1]).abs();
            // 在同一行高容差范围内 (8px) 优先按 X 轴排序
            if y_diff < 8.0 {
                a.coords[0].total_cmp(&b.coords[0])
            } else {
                a.coords[1].total_cmp(&b.coords[1])
            }
        });
        sorted
    }

    /// 基于 X 轴投影直方图检测是否存在中缝空白谷底
    fn detect_vertical_valley(items: &[TextBoxItem], page_width: u32) -> Option<(f32, f32)> {
        let bins = 100;
        let mut hist = vec![0usize; bins];
        let bin_width = page_width as f32 / bins as f32;

        if bin_width <= 0.0 {
            return None;
        }

        // 仅对页面中部 30% ~ 70% 的文本进行投影统计
        for item in items {
            let x1 = (item.coords[0] / bin_width) as usize;
            let x2 = ((item.coords[2] / bin_width) as usize).min(bins - 1);
            for b in x1..=x2 {
                hist[b] += 1;
            }
        }

        // 在 35% ~ 65% 区域内寻找连续为 0 或极小值的谷底
        let start_bin = (bins as f32 * 0.35) as usize;
        let end_bin = (bins as f32 * 0.65) as usize;

        let mut max_zero_len = 0;
        let mut best_start = 0;
        let mut current_zero_len = 0;
        let mut current_start = 0;

        for b in start_bin..=end_bin {
            if hist[b] <= 1 {
                if current_zero_len == 0 {
                    current_start = b;
                }
                current_zero_len += 1;
            } else {
                if current_zero_len > max_zero_len {
                    max_zero_len = current_zero_len;
                    best_start = current_start;
                }
                current_zero_len = 0;
            }
        }

        if current_zero_len > max_zero_len {
            max_zero_len = current_zero_len;
            best_start = current_start;
        }

        // 谷底宽度超过页面宽度的 3% (即 >= 3 bins)
        if max_zero_len >= 3 {
            let gap_x1 = best_start as f32 * bin_width;
            let gap_x2 = (best_start + max_zero_len) as f32 * bin_width;
            Some((gap_x1, gap_x2))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_column_sort() {
        let items = vec![
            TextBoxItem { text: "B".into(), score: 0.9, coords: [10.0, 50.0, 100.0, 70.0] },
            TextBoxItem { text: "A".into(), score: 0.9, coords: [10.0, 10.0, 100.0, 30.0] },
        ];
        let sorted = ReadingOrder::sort_reading_order(&items, 500);
        assert_eq!(sorted[0].text, "A");
        assert_eq!(sorted[1].text, "B");
    }
}
