use anyocr::{Engine, EngineConfig, DocumentFormat};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== anyocr 极简使用示例 ===");

    let sample_image = "/Users/icychick/.gemini/antigravity-cli/brain/6efdb7c6-eaca-4a22-8acc-97baeea17ede/.user_uploaded/uploaded_media_1789028772708.png";
    if !Path::new(sample_image).exists() {
        println!("未检测到样本图片，示例退出");
        return Ok(());
    }

    let bytes = std::fs::read(sample_image)?;

    // 1. 极简自由函数 (类似 anydoc，开箱即用)
    println!("-> 方式一：顶层开箱即用接口 (anyocr::to_markdown)");
    let md = anyocr::to_markdown(&bytes, DocumentFormat::Auto)?;
    println!("解析产出 Markdown 长度: {} 字符", md.len());

    // 2. 显式持有 Engine 实例 (高吞吐/多任务复用推荐)
    println!("\n-> 方式二：工业级显式引擎实例 (Engine::new)");
    let engine = Engine::new(EngineConfig::default())?;
    let doc = engine.parse(&bytes, DocumentFormat::Auto)?;

    println!("总页数: {}, 解析耗时: {}ms", doc.total_pages, doc.elapsed_ms);
    for page in &doc.pages {
        println!("第 {} 页包含 {} 个文字框, {} 个 AST 语法块", page.page_index + 1, page.boxes.len(), page.blocks.len());
    }

    println!("\n--- 最终 Markdown 内容预览 ---");
    println!("{}", doc.markdown);

    Ok(())
}
