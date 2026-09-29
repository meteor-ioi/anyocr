use anyocr::{Engine, EngineConfig, ModelProfile};
use std::path::PathBuf;
use std::time::Instant;

#[repr(C)]
struct RUsage {
    ru_utime: [i64; 2],
    ru_stime: [i64; 2],
    ru_maxrss: i64,
    _pad: [i64; 13],
}

unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut RUsage) -> i32;
}

fn get_peak_memory_mb() -> f64 {
    unsafe {
        let mut usage: RUsage = std::mem::zeroed();
        getrusage(0, &mut usage);
        usage.ru_maxrss as f64 / (1024.0 * 1024.0)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pdf_dir = PathBuf::from("/Users/icychick/Desktop/PDF");
    let cache_img_dir = pdf_dir.join("_page_images");
    let out_dir = pdf_dir.join("md_anyocr_rust");
    std::fs::create_dir_all(&out_dir)?;

    let target_pdfs = vec![
        ("Cortina PO-004659.pdf", 20),
        ("KOSJ20260129-6.pdf", 14),
        ("PC-PO.2606034.pdf", 1),
        ("SAFETY 091175 - 091181 Rev.pdf", 7),
        ("Safety Jogger April 2026.pdf", 1),
        ("SOX73836627DB986193.PDF", 2),
        ("SOX8A831BC63F491BA2.PDF", 2),
    ];

    println!("======================================================================");
    println!("🚀 AnyOCR 原生 Rust 双轨流水线 (PicoDet-S + PP-OCRv6 + SLANet+) 基准实测");
    println!("======================================================================");

    let config = EngineConfig {
        profile: ModelProfile::Fast,
        enable_table: true,
        ..Default::default()
    };

    let t_init = Instant::now();
    let engine = Engine::new(config)?;
    println!(
        "✅ 引擎初始化就绪！耗时: {:.2}s | 初始物理内存: {:.1} MB\n",
        t_init.elapsed().as_secs_f64(),
        get_peak_memory_mb()
    );

    let total_start = Instant::now();
    let mut total_pages_processed = 0;
    let mut total_tables_detected = 0;
    let mut total_text_boxes = 0;

    println!("======================================================================");
    println!("开始逐页评测 7 份真实工业订单原件图像 (共 47 页)");
    println!("======================================================================");

    for (idx, (filename, expected_pages)) in target_pdfs.iter().enumerate() {
        let stem = filename.trim_end_matches(".pdf").trim_end_matches(".PDF");
        println!("\n📂 [{}/7] 正在处理文档: {} (共 {} 页)", idx + 1, filename, expected_pages);

        let doc_start = Instant::now();
        let mut doc_markdown_pages = vec![format!("# {}\n\n*生成引擎: AnyOCR 原生 Rust 双轨版面识别 (PicoDet-S + PP-OCRv6 + SLANet+)*\n", stem)];
        let mut doc_tables = 0;
        let mut doc_boxes = 0;

        for p in 1..=*expected_pages {
            let img_path = cache_img_dir.join(format!("{}_p{}.png", stem, p));
            if !img_path.exists() {
                println!("  ⚠️ 未找到页面图像: {}", img_path.display());
                continue;
            }

            let p_img = image::open(&img_path)?;
            let p_start = Instant::now();
            let parsed = engine.parse_image(&p_img)?;
            let p_elapsed = p_start.elapsed().as_secs_f64();

            let mut page_tables = 0;
            let mut page_boxes = 0;
            for page in &parsed.pages {
                page_boxes += page.boxes.len();
                for block in &page.blocks {
                    if matches!(block, anyocr::DocBlock::Table { .. }) {
                        page_tables += 1;
                    }
                }
            }

            doc_tables += page_tables;
            doc_boxes += page_boxes;
            doc_markdown_pages.push(format!("### 第 {} 页\n\n{}\n", p, parsed.markdown));

            println!(
                "  - 第 {:>2}/{:>2} 页: 耗时 {:>4.2}s | 表格: {:>1} 个 | 文字框: {:>3} 个 | 内存: {:>6.1} MB",
                p, expected_pages, p_elapsed, page_tables, page_boxes, get_peak_memory_mb()
            );
        }

        let doc_elapsed = doc_start.elapsed().as_secs_f64();
        let full_md = doc_markdown_pages.join("\n\n---\n\n");
        let out_md_path = out_dir.join(format!("{}.md", stem));
        std::fs::write(&out_md_path, &full_md)?;

        let avg_page_sec = doc_elapsed / *expected_pages as f64;
        println!(
            "  ✨ 文档完成！总耗时: {:>5.2}s (平均 {:>4.2}s/页) | 表格: {:>2} 张 | 文字框: {:>4} | Markdown 已输出",
            doc_elapsed, avg_page_sec, doc_tables, doc_boxes
        );

        total_pages_processed += expected_pages;
        total_tables_detected += doc_tables;
        total_text_boxes += doc_boxes;
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    let peak_rss = get_peak_memory_mb();

    println!("\n======================================================================");
    println!("🎉 AnyOCR 原生 Rust 批量基准实测全部完成！统计报告如下：");
    println!("======================================================================");
    println!("• 处理文档总数    : {} 份", target_pdfs.len());
    println!("• 处理页面总数    : {} 页", total_pages_processed);
    println!("• 检出结构化表格  : {} 张", total_tables_detected);
    println!("• 提取识别文本框  : {} 个 (含物理绝对点坐标)", total_text_boxes);
    println!("• 纯 CPU 总耗时   : {:.2} 秒 ({:.1} 分钟)", total_elapsed, total_elapsed / 60.0);
    println!("• 全局平均单页速度: {:.2} 秒 / 页", total_elapsed / total_pages_processed as f64);
    println!("• 全程峰值物理内存: {:.1} MB", peak_rss);
    println!("======================================================================");

    Ok(())
}
