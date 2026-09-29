---
deadline: 2026-09-30
priority: high
tags: [optimization, rayon-pipelining, simd-resize, parallel-ocr, benchmark]
created: 2026-09-29
completed: 2026-09-29
---

# 多页 Rayon 流水线并发与切片 SIMD 预处理极速优化

### 📌 任务目标
探索并在纯 CPU 环境下将 47 页工业 PDF 耗时从 66.72 秒进一步压入极速区间，评估 SIMD 缩放与多 Worker 并发流水线的收益与代价，建立清晰的算力/内存选型基准。

### 📋 子任务清单 (Checklist)
- [x] 子任务 1：创建 Git 安全快照（`snapshot-before-pipelining` & `backup-before-pipelining`）
- [x] 子任务 2：实测 `fast_image_resize` 并排查微小切片负优化根因：在 48px 小图上滤波矩阵初始化开销远超双线性插值，及时回退保住基线
- [x] 子任务 3：实施基于 Rayon 的多 Worker 页面级并行流水线（支持 `ANYOCR_WORKERS` 动态调节）
- [x] 子任务 4：校准版面置信度至 0.40，物理单元格数大幅提升至 4,182 个（98% 对齐 Python 版 4,270）
- [x] 子任务 5：全量跑测 47 页工业 PDF：2-Worker 满核模式跑出 **54.05 秒**（单页 1.15s，逼近 Python 51s）；1-Worker 内存保护模式稳定在 **2.1GB**
- [x] 子任务 6：更新看板与文档索引

### 📝 开发记录与进度
- *2026-09-29 14:02*：创建安全快照 `snapshot-before-pipelining` 与备份分支 `backup-before-pipelining`；立项建卡。
- *2026-09-29 14:08*：引入 Rayon 2-Worker 并发流水线，47 页工业 PDF 耗时跑出 **54.05 秒**（单页 1.15s），单元格数达 **4,182 个**。
- *2026-09-29 14:16*：排查发现 `fast_image_resize` 在微小切片（48px）上存在动态加权表开销导致负优化，果断回退为极简矢量化 Triangle 插值；提供 `ANYOCR_WORKERS` 环境变量，支持在 54s 极速模式（4.1GB）与 2.1GB 安全模式之间灵活切换。
