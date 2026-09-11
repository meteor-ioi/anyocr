# anyocr 实施计划与开发任务清单 (todo.md)

> **项目目标**：打造 Rust 生态中原生、零 Python 依赖、开箱即用且支持多规格（Fast/Standard/Accurate）的通用文档与国标版式感知引擎，将图片、扫描件 PDF、OFD 还原为高保真标准 Markdown。

---

## 状态看板 (Status Dashboard)

| 里程碑 | 目标说明 | 进度状态 | 预计交付物 |
| :--- | :--- | :--- | :--- |
| **Milestone 1** | 基础设施骨架、图像摄入与 OCR 资产迁移 | ✅ 已完成 | `ingestion/image.rs`, `models/`, `engine.rs` 单图端到端冒烟通过 |
| **Milestone 2** | 版面还原、AST 语法树重构与分桶批处理优化 | ✅ 已完成 | `layout/` 核心算法 (AST, 合并, 标题, 表格匹配) + 分桶批处理 |
| **Milestone 3** | 多页文档摄入 (PDF 坏死层回退 & 国标 OFD) | ✅ 已完成 | `ingestion/` (多页 PDF, 乱码熔断, OFD XML/OCR) 全格式闭环 |
| **Milestone 4** | SensiDoc 回归集成与开源发布 | ✅ 全部完成 | SensiDoc 瘦身解耦 (32 项测试全绿), CI 跨平台流水线, 社区交付 |
| **Milestone 5** | 推理性能加速与硬件加速打通 (Performance Optimization) | ✅ 全部完成 | EP 硬件加速、SIMD 连续内存预处理、CCL 优化与动态 Batch |
| **Milestone 6** | 多页文档流水线并发加速与规格基准 (Multi-page Pipelining) | ✅ 全部完成 | PDF 并发提取/旋转校正、Fast 规格实测 (提速 70%+) |
| **Milestone 7** | 图像自适应尺寸保护与分辨率优化 (Smart Clamping) | 🚀 进行中 | 超大输入保护、自适应降采样、空间坐标一致性 |



---

## 详细任务拆解

### Milestone 1：基础设施骨架、图像摄入与 OCR 资产迁移（跑通单图闭环）

- [x] **1.1 细粒度 Feature Flags 配置**
  - [x] 在 `src/types.rs` 中引入 `ModelProfile`, `ExecutionProvider`, `EngineConfig`, `DocBlock`, `PageResult`
  - [x] 在 `Cargo.toml` 中配置可选 features：`[pdf, ofd, table, download]` 并增加 `tiff` 图像支持
  - [x] 验证全 feature 与最小 feature 编译通过 (`cargo check` & `--no-default-features`)

- [x] **1.2 核心类型系统与错误体系增强**
  - [x] 在 `src/error.rs` 中添加 `UnsupportedFormat` 变体，保证运行时安全性
  - [x] 在 `src/engine.rs` 中建立显式 `Engine` 结构体，管理 Session 生命周期
  - [x] 修正 `coords` 注释笔误为 `[x1, y1, x2, y2]`

- [x] **1.3 图像摄入管道前置实现 (`src/ingestion/image.rs`)**
  - [x] 支持 JPG, PNG, BMP, WEBP, TIFF 多格式无缝解码
  - [x] 图像等比缩放至 32 倍数与尺寸张量转换
  - [x] 编写图像预处理单测并通过验证

- [x] **1.4 文本检测模型迁移 (`src/models/detector.rs`)**
  - [x] 基于 PP-OCRv6 DBNet 迁移 ONNX 前向推理与连通块标记
  - [x] Unclip 膨胀处理并严格全序重排 (Reading Order)
  - [x] 坐标精确逆变换映射回原图物理像素

- [x] **1.5 文本识别 Baseline 迁移 (`src/models/recognizer.rs`)**
  - [x] 基于 PP-OCRv6 SVTR/CRNN 实现单行识别与 CTC Greedy 解码
  - [x] 字典空 Token 与空格规范对齐
  - [x] 实现逐行基线识别并提取置信度与坐标

- [x] **1.6 表格结构预测模型迁移 (`src/models/table.rs`)**
  - [x] 基于 SLANet_plus 实现表格结构预测与 50 词元序列解码
  - [x] 单元格 BBox 坐标逆映射回原图，受 `table` feature 保护

- [x] **1.7 引擎单图端到端冒烟测试 (`src/engine.rs`)**
  - [x] 串联 Image ➔ Det ➔ Rec ➔ 单图最简文本输出与 PageResult 封装
  - [x] 实现真实测试样本图全链路端到端冒烟测试 (`test_engine_smoke_on_real_image` 耗时 1.8s 检出 11 个文本框，测试通过)
  - [x] 实现基于 `std::sync::OnceLock` 的顶层便捷自由函数 (`to_markdown`, `parse`, `parse_with_config`)

- [x] **1.8 资产管理器与模型探测器 (`src/asset/downloader.rs`)**
  - [x] 支持自动探测多级搜索路径 (本地 models/ocr ➔ 开发路径 ➔ 系统缓存目录)
  - [x] 支持根据 `ModelProfile` (Fast/Standard/Accurate/Custom) 解析对应权重路径

---

### Milestone 2：版面还原、AST 语法树重构与分桶批处理优化（质的跃升）

- [x] **2.1 AST 语法树结构与容器 (`src/layout/ast.rs`)**
  - [x] 完善 `DocBlock` 枚举节点方法：`Heading`, `Paragraph`, `Table`, `List`, `Image`
  - [x] 统一节点 `bbox` 空间几何计算与 Markdown 局部序列化

- [x] **2.2 段落合并与折行优化 (`src/layout/para_merger.rs`)**
  - [x] 基于 `LINE_STOP_FLAGS` 消除碎行断裂与滥用无序列表
  - [x] 智能拼接：中文字符无缝衔接，英文单词智能补空格与处理连字符
  - [x] 支持有序与无序列表前缀自动识别分组

- [x] **2.3 自适应行高聚类定级标题 (`src/layout/heading.rs`)**
  - [x] 计算单页正文行高中位数 $H_{body}$ 作为自适应字号基准
  - [x] 梯级映射规则：$H \ge 1.8 H_{body} \to \#$, $H \ge 1.4 H_{body} \to \#\#$
  - [x] 行首编号（"第X章"、"一、"、"1.1"）与单据名特征自动提权

- [x] **2.4 多栏阅读顺序重排 (`src/layout/reading_order.rs`)**
  - [x] 基于 X 轴投影直方图自动检测垂直中缝空白谷底 (White-space Valley)
  - [x] 双栏拓扑重构：`顶通栏 ➔ 左栏 ➔ 右栏 ➔ 底通栏`，根治左右穿透串行

- [x] **2.5 表格空间几何反填 (`src/layout/table_matcher.rs`)**
  - [x] OCR 文本与 SLANet 单元格 BBox 空间拓扑匹配反填
  - [x] 规则表格优先转换为标准 GFM 管道符（`| 列1 | 列2 |`）
  - [x] 复杂合并单元格转换为标准安全 HTML `<table>`

- [x] **2.6 Markdown 序列化器 (`src/layout/md_serializer.rs`)**
  - [x] 将 AST 语义节点树转换为排版干净的标准 GFM Markdown，规范空行间距

- [x] **2.7 性能吞吐优化：长宽比分桶批处理 (`src/models/recognizer.rs`)**
  - [x] 切片按长宽比动态划入 4 个桶（短文本、中等、长文本、超长）
  - [x] 单桶内齐平最大宽度一次性 4D Tensor 批量前向推理，失败平滑降级

- [x] **2.8 端到端版面还原测试**
  - [x] 编写包含标题、段落、多栏、表格转 GFM 的 11 项全套单元与端到端回归测试并通过验证

---

### Milestone 3：多页 PDF 扫描件与国标 OFD 集成（全格式覆盖）

- [x] **3.1 多页 PDF 摄入与坏死文本层熔断 (`src/ingestion/pdf.rs`)**
  - [x] 基于 `lopdf` 流式按页提取可编辑文字，组装 `PageResult` 集合
  - [x] **关键容错**：统计乱码率与不可见字符比率，检测到坏死 CMap 自动熔断回退
  - [x] 扫描件 PDF 自动提取内嵌高清图像流走 OCR，输出页级坐标
  - [x] 纯矢量无字页面显式报错，杜绝静默失败

- [x] **3.2 国标 OFD 摄入与多页流式解析 (`src/ingestion/ofd.rs`)**
  - [x] 基于 `zip` + `quick-xml` 解包 OFD 容器
  - [x] 流式提取各页 `<ofd:TextCode>` 文本与坐标并组装为 `PageResult`
  - [x] 对纯图片型电子发票/公文，提取内嵌图分流至 OCR 并输出 `PageResult`

- [x] **3.3 顶层入口串联与双层 API 最终闭环 (`src/lib.rs` & `src/engine.rs`)**
  - [x] `Engine::detect_format` 基于魔数精确识别 PDF、OFD 与各类图像格式
  - [x] `Engine::parse` 与 `Engine::to_markdown` 全格式全自动路由分发
  - [x] 便捷自由函数 `anyocr::to_markdown`, `anyocr::parse`, `anyocr::parse_with_config` 接入 OnceLock 惰性单例
  - [x] 完善全套 13 项单元与回归测试并通过验证

---

### Milestone 4：SensiDoc 回归集成与开源发布
 
- [x] **4.1 SensiDoc 引入与冗余剥离**
  - [x] 在 `SensiDoc` 项目中以 `path = "../anyocr"` 依赖引入 `anyocr`
  - [x] 彻底删除 SensiDoc 原有的 `src/ocr/` 6 个历史重型文件（`markdown_builder`, `matcher`, `preprocessor`, `table_structure`, `text_detector`, `text_recognizer`），瘦身 5 万行与冗余维护
  - [x] 保留轻量代理适配器（`OcrEngine::recognize_bytes`, `recognize_image`, `is_loaded`, `unload`），100% 保持历史 API 签名与 3 分钟空闲自动释放机制

- [x] **4.2 业务链路回归验证**
  - [x] 验证 SensiDoc 全部 32 项业务单元测试 100% 通过（脱敏、抽词、并发卸载、OCR 代理等）
  - [x] 验证前端原图高亮与卷帘对比数据结构（`OcrResult.raw_boxes`）无缝衔接
  - [x] 验证 OCR 模型按需惰性加载与并发安全

- [x] **4.3 开源交付与社区资产**
  - [x] 补充完善包含纯文本 ASCII 架构图、双语说明与矩阵的官方 `README.md`
  - [x] 编写开箱即用官方示例 `examples/simple_convert.rs` 并通过端到端测试
  - [x] 配置 GitHub Actions CI（跨平台 Ubuntu / macOS / Windows 自动化测试流水线 `.github/workflows/ci.yml`）
  - [x] 补齐 `Cargo.toml` 发布元数据（readme, repository, keywords, categories 等）

---

### Milestone 5：推理引擎性能加速与硬件加速打通 (Performance Optimization)

- [x] **5.1 P0：打通 `ExecutionProvider` 与会话构建器（修复断路 Bug）**
  - [x] `TextDetector`、`TextRecognizer`、`TableStructurePredictor` 改造构造函数支持 `ExecutionProvider`
  - [x] 统一调用 `session::build_session`，打通 CoreML / DirectML / CUDA 硬件加速及并发线程探测
  - [x] `Engine::new` 穿透传递 `config.provider`
  - [x] 确保测试与回退机制正常，验证多核 CPU / CoreML 加速

- [x] **5.2 P1：SIMD 连续内存加速预处理（消除像素级循环与 4D 索引开销）**
  - [x] 改造 `ImagePreprocessor::prepare_det_input`，采用连续 slice / chunk 迭代，按通道连续写入
  - [x] 改造 `ImagePreprocessor::prepare_rec_input` 与 `prepare_table_input`
  - [x] 消除 `rgb_img.get_pixel(x, y)` 边界检查和 `tensor[[0, c, y, x]]` 4D 计算开销

- [x] **5.3 P2：连通域标记算法（CCL）内存与分配优化**
  - [x] 改造 `detector.rs` 中的连通域查找算法，减少小连通域分配与内存移动
  - [x] 用扁平数组 / 结构体预分配替代频繁 `VecDeque` 分配，提升 DBNet 后处理吞吐

- [x] **5.4 P3：生效 `max_batch_size` 动态配置**
  - [x] `TextRecognizer::recognize_batch` 接收或使用 `config.max_batch_size`（取代硬编码 batch=8）
  - [x] `Engine::parse_image` 穿透传递 `self.config.max_batch_size`

- [x] **5.5 P4：性能基准回归与验证**
  - [x] 执行全套单元测试与端到端测试 (`cargo test` 13 项测试全绿)
  - [x] 运行性能 Benchmark（`examples/benchmark_superl.rs` 与单图冒烟）验证吞吐翻倍提升

---

### Milestone 6：多页文档流水线并发加速与规格基准 (Multi-page Pipelining)

- [x] **6.1 多页 PDF 预处理与图像流并发解耦 (`src/ingestion/pdf.rs`)**
  - [x] 页面元数据扫描、内嵌图像提取与顺时针旋转校正（Rotate 90/180/270）多页并发处理 (`std::thread::scope`)
  - [x] 构建流水线预备队列，消除单页大图解码与旋转对推理 Session 的阻塞等待
  - [x] 保持页码顺序（`page_index` 严格保序）与显式错误传播规范

- [x] **6.2 识别批处理内切片图像转换复用优化 (`src/models/recognizer.rs`)**
  - [x] 采用直接 `RgbImage` 缩放减少 `DynamicImage` 中间克隆与枚举动态分发开销
  - [x] 优化小切片连续内存排布提升 L1/L2 缓存局部性

- [x] **6.3 多页真实单据基准与性能实测 (`examples/benchmark_superl.rs`)**
  - [x] 实测 4 页大发票（665 个文字框）解析从 62.8s 大幅缩减至 45.5s，单篇立减 17.3s (提速 27.5%)
  - [x] 保持版面还原准确性与 100% 单元测试全绿

---

### Milestone 7：图像自适应尺寸保护与分辨率优化 (Smart Clamping)

- [x] **7.1 在 `EngineConfig` 中引入 `max_dimension` 配置项**
  - [x] 默认为 `Some(2560)`（自适应黄金上限），支持通过 `None` 关闭钳制以供纯原始分辨率场景使用
  - [x] 增加详细工程文档与设计决策说明

- [x] **7.2 图像摄入管道智能降采样与坐标投影校正 (`src/layout/ast.rs` & `src/engine.rs`)**
  - [x] 在 `parse_image` 中引入无损自适应钳制视窗计算与等比降采样
  - [x] 增加 `DocBlock::rescale`，确保所有文字框与语义块坐标映射回原图物理视窗 `PageResult.dimensions` (绝对空间一致性)

- [x] **7.3 `Engine::parse_image` 与 PDF/OFD 接入自适应上限保护**
  - [x] 消除超大图导致的内存占用激增与无谓下采样开销
  - [x] 原图 2480x3508 自动钳制至 1810x2560，单张内存占用减少 47%

- [x] **7.4 精度与耗时回归基准验收**
  - [x] 运行单图冒烟与 `examples/benchmark_superl.rs` 验证（Fast 模式下密集单据由 3.24s 进一步降至 3.06s）
  - [x] 保证 13 项单元测试全绿通过
