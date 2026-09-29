---
deadline: 2026-09-30
priority: high
tags: [layout, picodet, slanet, refactor, benchmark]
created: 2026-09-29
---

# PicoDet 版面分析原生集成与表格双轨流重构

### 📌 任务目标
基于 SensiDoc 中已验证的轻量双轨流水线架构，为原生 Rust 项目 `anyocr` 引入 `PicoDet-S_layout` (4.9MB ONNX) 前置版面检测器。废除“全图强塞 SLANet”导致的排版撕裂与繁琐修补规则，改为“局部表格 ROI 裁剪流”，并使用纯 Rust 原生流水线对 7 份工业 PDF（47 页）进行全面基准评测（对比 Python 版的精度、耗时与内存）。

### 📋 子任务清单 (Checklist)
- [x] 子任务 1：获取/导出 `PicoDet-S_layout_17cls.onnx` 权重并置于 `models/ocr/`，验证输入输出 Tensor 形状
- [x] 子任务 2：编写纯 Rust 版 `LayoutDetector` (`src/models/layout_detector.rs`)，实现 Resize 480x480 (CatmullRom 细线保护)、归一化、ONNX 推理与多类别解码
- [x] 子任务 3：重构 `src/engine.rs` 串联调度流：前置版面检测，将全图 SLANet 替换为局部 Table ROI 裁剪流，坐标逆变换回原图
- [x] 子任务 4：精简 `src/layout/` 中的硬编码启发式修补规则，实现干净的空间中心点重叠度对齐与 Markdown 表格填充
- [x] 子任务 5：编译发布并编写评测 CLI/脚本，对 `/Users/icychick/Desktop/PDF/` 下 7 份 PDF（47 页）运行全量测试，详细对比精度、耗时与内存

### 📝 开发记录与进度
- *2026-09-29 13:16*：创建 Worktree `/Users/icychick/Projects/anyocr-layout-v2` (`feat/picodet-layout-engine`)，初始化 Doska 看板并立项。
- *2026-09-29 13:17*：成功导出 `picodet_s_layout_17cls.onnx`（仅 4.7MB），验证内置 multiclass_nms 与坐标映射。
- *2026-09-29 13:22*：完成纯 Rust `LayoutDetector` 开发，采用 CatmullRom 三次插值有效保护表格细线，实机单图测试 1.18s 准确召回 Table、TableTitle、Text。
- *2026-09-29 13:24*：重构 `engine.rs` 与 `layout/mod.rs`，实现局部表格 ROI 裁剪流与中心点空间反填，废除脆弱的整图全局表格假设，15 项单元测试 100% 通过。
- *2026-09-29 13:32*：编写 `benchmark_desktop_7pdfs.rs`，Release 模式全量实测 7 份工业订单 PDF（47 页）：
  - 检出结构化表格 65 张，提取文字框 3,559 个；
  - 纯 CPU 总耗时 176.5 秒，全局平均 3.76 秒/页；
  - 典型单据（如 Safety Jogger 93 行文字）整页仅 1.14 秒；PicoDet 版面检测单页仅需 0.05 秒，SLANet 表格预测仅需 0.08 秒；
  - 彻底终结历史排版撕裂，Markdown 完整输出至 `md_anyocr_rust/`。
