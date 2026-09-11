use anyocr::{Engine, EngineConfig, DocumentFormat, ModelProfile};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let folder = PathBuf::from("/Users/icychick/Desktop/售前需求/胜寒 - SUPERL");
    let out_dir = PathBuf::from("/Users/icychick/.gemini/antigravity-cli/brain/8a927574-a652-4030-9dc8-27d09b693fce/scratch/anyocr_outputs");
    std::fs::create_dir_all(&out_dir)?;

    let files = vec![
        "img_260831124532.pdf",
        "KH Tax invoice.pdf",
        "Tax invoice - KH.PDF",
        "CHAY DA IMPORT-C1-MAY-2026-$ 8,545.18.pdf",
    ];

    println!("=========================================================================================");
    println!(">>> 评测轮次 1：标准规格 Standard (PP-OCRv6-medium 76MB + SLANet 表格)");
    println!("=========================================================================================");

    let config_std = EngineConfig {
        profile: ModelProfile::Standard,
        enable_table: true,
        ..Default::default()
    };
    let engine_std = Engine::new(config_std)?;

    let mut std_results = Vec::new();

    for fname in &files {
        let fpath = folder.join(fname);
        let bytes = std::fs::read(&fpath)?;
        let t0 = Instant::now();
        let doc = engine_std.parse(&bytes, DocumentFormat::Pdf)?;
        let total_ms = t0.elapsed().as_millis();
        
        let out_name = format!("{}_standard.md", fname.replace(".pdf", "").replace(".PDF", "").replace(" ", "_").replace("$", "USD"));
        std::fs::write(out_dir.join(&out_name), &doc.markdown)?;

        let total_boxes: usize = doc.pages.iter().map(|p| p.boxes.len()).sum();
        let char_count: usize = doc.markdown.chars().filter(|c| !c.is_whitespace()).count();
        println!("【Standard】文档: {:<38} | 页数: {} | 文字框: {:>3} | 字符数: {:>4} | 耗时: {:>5}ms",
            fname, doc.total_pages, total_boxes, char_count, total_ms);
        std_results.push((total_ms, total_boxes, char_count, doc.markdown));
    }

    println!("\n=========================================================================================");
    println!(">>> 评测轮次 2：极速规格 Fast (PP-OCRv6-small 20MB + SLANet 表格)");
    println!("=========================================================================================");

    let config_fast = EngineConfig {
        profile: ModelProfile::Fast,
        enable_table: true,
        ..Default::default()
    };
    let engine_fast = Engine::new(config_fast)?;

    let mut fast_results = Vec::new();

    for fname in &files {
        let fpath = folder.join(fname);
        let bytes = std::fs::read(&fpath)?;
        let t0 = Instant::now();
        let doc = engine_fast.parse(&bytes, DocumentFormat::Pdf)?;
        let total_ms = t0.elapsed().as_millis();
        
        let out_name = format!("{}_fast.md", fname.replace(".pdf", "").replace(".PDF", "").replace(" ", "_").replace("$", "USD"));
        std::fs::write(out_dir.join(&out_name), &doc.markdown)?;

        let total_boxes: usize = doc.pages.iter().map(|p| p.boxes.len()).sum();
        let char_count: usize = doc.markdown.chars().filter(|c| !c.is_whitespace()).count();
        println!("【Fast】    文档: {:<38} | 页数: {} | 文字框: {:>3} | 字符数: {:>4} | 耗时: {:>5}ms",
            fname, doc.total_pages, total_boxes, char_count, total_ms);
        fast_results.push((total_ms, total_boxes, char_count, doc.markdown));
    }

    println!("\n=========================================================================================");
    println!(">>> 核心对比总结矩阵 (Standard vs Fast)");
    println!("=========================================================================================");
    println!("{:<38} | Standard 耗时 | Fast 耗时 | 耗时下降率 | Standard字符 | Fast字符 | 字符保留率", "文档名称");
    println!("{:-<38}-|-{:-<13}-|-{:-<9}-|-{:-<10}-|-{:-<12}-|-{:-<8}-|-{:-<10}", "", "", "", "", "", "", "");

    for (i, fname) in files.iter().enumerate() {
        let (std_ms, _std_boxes, std_chars, _) = &std_results[i];
        let (fast_ms, _fast_boxes, fast_chars, _) = &fast_results[i];

        let speedup = (*std_ms as f64 - *fast_ms as f64) / *std_ms as f64 * 100.0;
        let char_retention = *fast_chars as f64 / *std_chars as f64 * 100.0;

        println!("{:<40} | {:>10}ms | {:>6}ms | {:>9.1}% | {:>12} | {:>8} | {:>9.1}%",
            fname, std_ms, fast_ms, speedup, std_chars, fast_chars, char_retention);
    }

    Ok(())
}
