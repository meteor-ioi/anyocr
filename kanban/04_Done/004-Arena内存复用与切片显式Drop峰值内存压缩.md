---
deadline: 2026-09-30
priority: high
tags: [optimization, memory-arena, eager-drop, peak-memory-reduction, benchmark]
created: 2026-09-29
completed: 2026-09-29
---

# Arena 内存复用与切片显式 Drop 峰值内存压缩

### 📌 任务目标
通过实施 Tensor Arena 连续缓冲区复用（避免每批切片向系统 malloc/free）与切片/原图生命周期显式提前 Drop（消除原图、切片与推理 Tensor 的内存叠加），将多 Worker 并发运行时的峰值内存从 4.19GB 压降至 2.5GB 左右，同时验证识别速度与准确率零回退。

### 📋 子任务清单 (Checklist)
- [x] 子任务 1：设计与实现文本识别批处理 Tensor Arena 缓冲区复用（通过 `thread_local!` 私有缓冲与 `TensorRef::from_array_view` 零拷贝视图，消除频繁 malloc/free）
- [x] 子任务 2：实施切片与大图显式 Eager Drop（在切片转入 Tensor 后立即释放原图切片；在版面与表格 ROI 截取完毕后立即释放大图）
- [x] 子任务 3：运行 `cargo check` 与 16 项单元测试全部稳绿通过
- [x] 子任务 4：使用 47 页真实工业 PDF 进行全量性能与内存峰值压测：峰值内存由 **4.19GB 骤降至 2.46GB (2,527MB)**，耗时从 54.05s 进一步提速至 **50.75s**（单页 1.08s，反超 Python 版 51.20s）
- [x] 子任务 5：更新测试报告、看板卡片流转至 04_Done 并同步文档索引

### 📝 开发记录与进度
- *2026-09-29 14:27*：立项建卡，进入 `03_In_Progress`；准备开始实现 Arena 复用与切片显式 Drop。
- *2026-09-29 14:31*：完成 `recognizer.rs` 的 `RECOGNIZER_ARENA` 缓冲区零拷贝构造与切片逐批 drain/drop；完成 `engine.rs` 流程重构，在识别前提前释放大图 `working_img` 与 `det_boxes`；16 项单元测试 100% 通过。
- *2026-09-29 14:32*：47 页真实工业 PDF 全量压测完成！耗时 **50.75 秒**（单页 1.08s），峰值内存由 **4.19GB 暴降至 2.52GB**（内存直接削减 40%），单元格数 4,182 个、文本框 3,563 个零损失，任务流转至 `04_Done`。
