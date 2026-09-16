use anyocr::{Engine, EngineConfig, DocumentFormat};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/icychick/Pictures/CHAY DA IMPORT-C1-MAY-2026-$ 8,545.18_1.png".to_string()
    });

    println!("正在分析图像: {}", img_path);
    if !Path::new(&img_path).exists() {
        eprintln!("图像不存在!");
        return Ok(());
    }

    let bytes = std::fs::read(&img_path)?;
    let engine = Engine::new(EngineConfig::default())?;

    let doc = engine.parse(&bytes, DocumentFormat::Image)?;
    println!("\n=== 文档统计 ===");
    println!("总页数: {}, 耗时: {}ms", doc.total_pages, doc.elapsed_ms);
    
    for (p_idx, page) in doc.pages.iter().enumerate() {
        println!("\n--- 第 {} 页 (尺寸: {}x{}, 检测框数: {}, AST块数: {}) ---", 
            p_idx + 1, page.dimensions.0, page.dimensions.1, page.boxes.len(), page.blocks.len());
        
        for (b_idx, block) in page.blocks.iter().enumerate() {
            match block {
                anyocr::DocBlock::Heading { level, text, bbox } => {
                    println!("[Block {}] Heading(h{}): '{}' (bbox: {:?})", b_idx, level, text, bbox);
                }
                anyocr::DocBlock::Paragraph { lines, bbox } => {
                    println!("[Block {}] Paragraph: {} 行 (bbox: {:?})", b_idx, lines.len(), bbox);
                    for (l_idx, line) in lines.iter().enumerate() {
                        println!("    L{}: '{}' (bbox: {:?})", l_idx, line.text, line.coords);
                    }
                }
                anyocr::DocBlock::Table { markdown_table, raw_html, bbox } => {
                    println!("[Block {}] Table (bbox: {:?}):", b_idx, bbox);
                    println!("    GFM Table:\n{}", markdown_table);
                    if let Some(html) = raw_html {
                        println!("    Raw HTML:\n{}", html);
                    }
                }
                anyocr::DocBlock::List { ordered, items, bbox } => {
                    println!("[Block {}] List(ordered={}): {} 项 (bbox: {:?})", b_idx, ordered, items.len(), bbox);
                }
                anyocr::DocBlock::Image { format, alt, bbox } => {
                    println!("[Block {}] Image({}): {:?} (bbox: {:?})", b_idx, format, alt, bbox);
                }
            }
        }
    }

    println!("\n=== 输出 Markdown 全文 ===");
    println!("{}", doc.markdown);

    Ok(())
}
