# SAM3 项目开发日志

> 记录本项目中自定义 Python 程序的开发过程、设计思路与版本迭代。
> 日期以文件系统时间戳（LastWriteTime / CreationTime）为准；仓库暂未提交 git，后续接入版本控制后可补充 commit 记录。

---

## 一、总览

| 程序 | 作用 | 规模 | 最后修改 |
|------|------|------|----------|
| `sam3_video_text.py` | 视频分割·快速版（纯文本提示，跳过可视化） | 163 行 | 2026-07-08 |
| `sam3_video_point.py` | 视频分割·文本+点提示组合版 | 255 行 | 2026-07-08 |
| `sam3_video_box.py` | 视频分割·文本+框提示组合版 | 252 行 | 2026-07-22 |
| `sam3_video_double_point.py` | 视频分割·批量版（首尾双帧+Excel，命名与实现不符） | 372 行 | 2026-08-01 |
| `sam3_video_double_box.py` | 视频分割·批量版（首尾双帧+Excel 框坐标） | 372 行 | 2026-08-01 |
| `main-first copy.py` | 视频水位线自动化处理·v1 基准版 | 623 行 | 2026-08-05 |
| `main-first-with-2-frames.py` | 视频水位线·首尾双关键帧版 | 762 行 | 2026-08-11 |
| `main-first-with-3-frames.py` | 视频水位线·首中尾三关键帧版 | 762 行 | 2026-08-11 |
| `main-first-with-n-to-3-frames.py` | 视频水位线·n→3 帧版（IQR 剔异常+时序择优） | 836 行 | 2026-08-12 |

> ⚠️ 日期说明：所有文件 CreationTime 均为 2026-08-05（目录整体拷贝时间），实际开发修改以 LastWriteTime 为准。

---

## 二、版本演进关系

```
sam3_video_text.py (07-08)          ← 最早：纯文本快速分割
        │
        ├── sam3_video_point.py (07-08)   文本 + 点提示
        └── sam3_video_box.py   (07-22)   文本 + 框提示
                │
                └── sam3_video_double_box.py / double_point.py (08-01)
                                             批量版：首尾双帧 + Excel 坐标

main-first copy.py (08-05)          ← 单帧图像识别 → 生成框提示 → video 传播
        │
        ├── main-first-with-2-frames.py (08-11)  首/尾关键帧
        └── main-first-with-3-frames.py (08-11)  首/中/尾关键帧
                │
                └── main-first-with-n-to-3-frames.py (08-12)  ← 当前主线
                                    等间距抽 n 帧 → IQR 剔异常 → 择优 3 帧
```

---

## 三、`sam3_video_text.py` — 视频分割·快速版

**最后修改：2026-07-08 | 163 行**

### 功能概述
SAM3 视频分割的最早期版本。跳过 matplotlib 可视化，直接输出叠加掩码的结果视频。纯文本提示（`"water"`），无框/点辅助。

### 设计思路
最小可行流程：加载模型 → 读取全部帧 → 启动 session → 第 0 帧加文本提示 → 全片传播 → 逐帧叠加红色掩码 → 以 MP4 写出。

### 核心模块与代码要点
- `propagate_in_video` (`sam3_video_text.py:20`) — 流式传播，按 `frame_index` 收集结果。
- `read_video_frame` (`sam3_video_text.py:33`) — 用 `cv2.VideoCapture` 逐帧读取。
- `overlay_mask` (`sam3_video_text.py:46`) — 合并所有目标掩码为一张 bool 图，红色 `[0,0,255]` 按 0.6/0.4 混合叠加。
- `save_frames_as_mp4` (`sam3_video_text.py:67`) — 用 `mp4v` 编码写出，FPS **写死为 30**（未读原生帧率，是后期版本修复点）。

### 版本迭代 / 修复记录
| 日期 | 内容 |
|------|------|
| 2026-07-08 | 初版：完成纯文本提示的视频分割与结果输出 |

### 使用说明
`MODEL_PATH` / `VIDEO_PATH` / `OUTPUT_VIDEO` / `PROMPT_TEXT` / `PROMPT_FRAME` / `FPS` 在 `__main__` 配置区（`sam3_video_text.py:91-99`）修改。

---

## 四、`sam3_video_point.py` — 视频分割·文本+点提示组合版

**最后修改：2026-07-08 | 255 行**

### 功能概述
在文本提示基础上增加**点提示**（可多点正/负样本），用于更精准地指定水体的位置与范围。点坐标归一化，`POINT_COORDS = [[0.5, 0.8]]`，`POINT_LABELS = [1]`。

### 设计思路
复用快速版 6 步流程，将 `add_prompt` 请求中新增 `point_coords` / `point_labels` 字段。同时在首帧掩码兜底策略上保留"传播结果为空则用首帧静态掩码"的降级方案。

### 核心模块与代码要点
- `overlay_mask` (`sam3_video_point.py:60`) — 增加 `mask_bool.sum()==0` 的空掩码防御。
- 提示注入 (`sam3_video_point.py:176-183`) — `add_prompt` 同时携带 `text` 与 `point_coords`。
- 空掩码兜底 (`sam3_video_point.py:230-238`) — 传播结果为空时回退首帧静态掩码。

### 版本迭代 / 修复记录
| 日期 | 内容 |
|------|------|
| 2026-07-08 | 初版：由纯文本改为文本+点提示组合 |

### 使用说明
点坐标为归一化 `[[x, y], ...]`（0~1，左上角为原点），标签 `1`=正样本、`0`=负样本（`sam3_video_point.py:127-132`）。其余配置同快速版。

---

## 五、`sam3_video_box.py` — 视频分割·文本+框提示组合版

**最后修改：2026-07-22 | 252 行**

### 功能概述
在文本提示基础上增加**框提示**（`BOX_PROMPTS = [[0.5, 0.685, 1.0, 0.63]]`，`BOX_LABELS = [1]`），限定分割区域，提升水体分割稳定性。

### 设计思路
流程与点提示版一致，仅将 `point_coords` 换成 `bounding_boxes` / `bounding_box_labels`。**修复了 numpy 数组 `if probs:` 直接布尔判断抛 ValueError 的问题**，改用 `len(probs) > 0`。输出帧率由写死 30 改为读取视频原生 `CAP_PROP_FPS`。

### 核心模块与代码要点
- 原生帧率读取 (`sam3_video_box.py:48-59`) — `read_video_frames` 返回 `(video_frames, real_fps)`。
- 置信度打印修复 (`sam3_video_box.py:190-191`) — `if len(probs) > 0` 替代直接判数组。
- 结果视频按原生 FPS 写出 (`sam3_video_box.py:244`)。
- 首帧 PNG 落盘 `water-flow(1).png` (`sam3_video_box.py:239-241`)。

### 版本迭代 / 修复记录
| 日期 | 内容 |
|------|------|
| 2026-07-22 | 增加框提示；修复 numpy 布尔判断 ValueError；输出采用原生帧率 |

### 使用说明
框坐标为归一化 `[x, y, w, h]`，`1` 表示正样本（`sam3_video_box.py:127-130`）。提供 `PROMPT_FRAME` 指定提示帧（默认 0）。

---

## 六、`sam3_video_double_box.py` / `sam3_video_double_point.py` — 批量版（首尾双帧+Excel）

**最后修改：2026-08-01 | 各 372 行**

### 功能概述
将单视频处理升级为**批量流水线**：遍历输入目录内所有 `.mp4`，从两个 Excel 表分别读取**首帧框坐标**（Excel1）与**末帧框坐标**（Excel2），在首/尾两帧注入 文本+框 提示，再全片传播。

### 设计思路
- Excel 按文件名匹配，`第3列=上Y、第4列=下Y`，换算成归一化框坐标（`load_box_from_excel`）。
- 模型只加载一次，批量循环复用（`sam3_video_double_box.py:332`）。
- 首尾双帧提示策略：水线在视频首尾变化最大，两端约束能稳定跟踪。
- 失败视频单独记录、继续处理，最后汇总成功/失败统计。

### 核心模块与代码要点
- `load_box_from_excel` (`sam3_video_double_box.py:127`) — 读 Excel → 按 `filename` 匹配 → 生成归一化框。
- `process_single_video` (`sam3_video_double_box.py:157`) — 5 步：读视频/读 Excel/启 session/双帧加提示/传播+生成。
- 首帧掩码兜底 (`sam3_video_double_box.py:255-257`) — 无传播结果时用首帧掩码。
- 批量主循环 (`sam3_video_double_box.py:340-360`) — 单模型复用 + try/except 逐视频容错。

### ⚠️ 已知问题（如实记录）
`sam3_video_double_point.py` 与 `sam3_video_double_box.py` **内容完全相同**（`Compare-Object` 无差异），其实现实际使用 `bounding_boxes` 框提示而非点提示，与文件名"point"不符。属命名/拷贝遗留，后期如需点提示版应单独实现。

### 版本迭代 / 修复记录
| 日期 | 内容 |
|------|------|
| 2026-08-01 | 批量版初版：首尾双帧 + Excel 框坐标；修复 numpy 布尔判断；原生帧率输出；路径前置校验 |

### 使用说明
配置区 (`sam3_video_double_box.py:277-290`)：`INPUT_DIR` / `OUTPUT_DIR` / `EXCEL1_PATH`(首帧) / `EXCEL2_PATH`(末帧) / `PROMPT_TEXT` / `BOX_LABELS`。模型与 Excel 存在性前置校验（`sam3_video_double_box.py:292-304`）。

---

## 七、`main-first copy.py` — 视频水位线自动化处理·v1 基准版

**最后修改：2026-08-05 | 623 行**

### 功能概述
**"以图引频"流水线的开端**：先用 SAM3 **图像模型**对视频的 首/中/尾 三帧分别做单帧水位线识别，把识别出的水位线转成归一化框提示，再用 **视频模型** 全片传播得到逐帧水体分割与水位线。这是后续所有 `main-first-*` 版本的共同祖先。

### 设计思路
核心创新在于**先图后频两段式**：
1. 图像模型（轻量、单帧）确定"水面/水位线"在哪 → 生成 Prompt；
2. 释放图像模型，换视频模型基于这些 Prompt 全片跟踪传播；
3. 显存紧张是主要约束 → 每一步都释放模型、回收 CUDA 缓存。

### 核心模块与代码要点
- `get_box_prompts_from_frames` (`main-first copy.py:385`) — 对 首/中/尾 三帧分别调 `process_single_image(prompt="black water")`，返回 `waterline_y`、归一化框 `[[0.5, y_norm, 1.0, h_norm]]`、mask、bbox、score。
- 关键掩码函数：`get_mask_bottom_y`、`get_all_masks_sorted_by_confidence`、`remove_top_connected_components`、`keep_largest_connected_component`、`apply_morphology`。
- tracker-only 优先 + detector 回退 (`main-first copy.py:497-536`) — 先 `init_video_from_single_frame_seeds` 用单帧 mask 初始化 tracker；失败则回退 `add_prompts`（`water` + 框提示）走原 detector 流。
- `batch_process_videos_auto` (`main-first copy.py:432`) — **单进程循环**批量处理，逐视频 `del` 模型 + `empty_cache`。
- 目标图保存到桌面 `desktop_dir`（后期版本改为参数化 `save_target_folder`）。

### 版本迭代 / 修复记录
| 日期 | 内容 |
|------|------|
| 2026-08-05 | v1 基准版：以图引频（首/中/尾单帧识别→框提示→视频传播）；tracker-only+detector 双流回退；单进程批量 |

### 使用说明
路径配置在 `main()` 的 `paths` 字典（`main-first copy.py:600-`）：`input_video_folder` / `output_video_folder`。无子进程隔离，多视频长跑易积累显存碎片。

---

## 八、`main-first-with-2-frames.py` / `main-first-with-3-frames.py` — 关键帧参数化版

**最后修改：2026-08-11 | 各 762 行**

### 功能概述
在 v1 基础上将"关键帧"参数化：`num_key_frames=2`（首+尾）或 `3`（首+中+尾）。两个文件仅 `num_key_frames` 默认值（2 vs 3）与 `get_box_prompts_from_frames` 内帧集合不同，其余实现完全一致。

### 设计思路（相对 v1 的改进）
1. **关键帧可配置**：由写死的首/中/尾改为 `num_key_frames` 参数驱动（`main-first-with-2-frames.py:444-473`）。
2. **显存治理强化**：新增 `log_gpu_memory` 全程打点；finally 中 `_ALL_INFERENCE_STATES.clear()` 兜底清理类级会话、`model.cpu()` 强制释放 CUDA、`reset_peak_memory_stats()`。
3. **子进程隔离**：新增 `_video_worker` + `batch_process_videos_auto`（`spawn` 模式），每个视频在独立子进程中处理，进程退出后显存由 OS 彻底回收，解决长跑累积问题。
4. 目标图保存目录参数化 `save_target_folder`。

### 核心模块与代码要点
- `process_single_video(v_path, out_path, v_name, checkpoint_path, save_target_folder, num_key_frames=2)` (`main-first-with-2-frames.py:444`)。
- 关键帧计算 (`main-first-with-2-frames.py:465-473`) — `3` 时 `{0, mid, last}`；否则 `{0, last}`。
- 完整资源清理 (`main-first-with-2-frames.py:692-738`) — `close_session` → 清 `_ALL_INFERENCE_STATES` → 模型移 CPU → del 局部对象 → GC → `empty_cache` + `reset_peak_memory_stats`。
- 子进程隔离 (`main-first-with-2-frames.py:684-806`) — `mp.get_context("spawn")` 逐视频 `start/join`，`p.exitcode` 判断成败。

### 两版本差异对照
| 项目 | with-2-frames | with-3-frames |
|------|---------------|----------------|
| `num_key_frames` | 2（首+尾） | 3（首+中+尾） |
| 单帧识别帧集 | `{0, last}` | `{0, mid, last}` |
| 输入目录 | `original_video` | `video_segment_with_problem` |

### 版本迭代 / 修复记录
| 日期 | 内容 |
|------|------|
| 2026-08-11 | 关键帧参数化（2/3 帧可选）；子进程隔离防显存累积；`log_gpu_memory` 打点；`_ALL_INFERENCE_STATES`/`model.cpu()` 兜底清理 |

### 使用说明
路径在 `main()` 的 `paths`（`main-first-with-2-frames.py:736-`）；`num_key_frames` 在 `_video_worker` 调用处硬编码（`main-first-with-2-frames.py:675`）。

---

## 九、`main-first-with-n-to-3-frames.py` — n→3 帧版（当前主线）

**最后修改：2026-08-12 | 836 行**

### 功能概述
当前水位线处理的主线程序。将"固定关键帧"升级为**等间距抽 n 帧 → IQR 四分位剔异常 → 时序择优保留 3 帧**的两阶段筛选。n 默认 5，`main()` 中配置为 7。解决了单帧误检（如整帧分割失败导致水位线 Y 异常跳变）拖垮整个视频跟踪的问题。

### 设计思路
相比固定首/中/尾，多抽几帧能覆盖更多水面波动信息，但引入误检风险，因此加入统计筛选：
1. **IQR 剔异常**：对 n 帧水位线 Y 序列算 Q1/Q3/IQR，剔除落在 `[Q1-1.5·IQR, Q3+1.5·IQR]` 外的帧；
2. **兜底补回**：剔除后不足 3 帧时，按"离正常区间最近"从异常帧补回；
3. **时序择优**：剩余帧中取 最早/中间/最晚 三个位置，保证时序覆盖（尤其拟合后取中点折半）。

### 核心模块与代码要点
- `get_box_prompts_from_frames(frames, image_model, frame_indices)` (`main-first-with-n-to-3-frames.py:387`) — 帧集由外部传入，逐帧单帧识别，返回 Y/框/mask/bbox/score。
- `filter_n_to_best_3` (`main-first-with-n-to-3-frames.py:433`) — 核心筛选函数：
  - IQR 区间计算 `:444-449`（`np.percentile`，`iqr_factor=1.5`）；
  - 判定表打印 `:455-468`（超上限/低下限精确到 px）；
  - 不足 3 帧兜底补回 `:470-482`；
  - 时序择优 `[0, len//2, -1]` `:484-489`。
- 子进程隔离批量 `batch_process_videos_auto(..., n_sample_frames=5)` (`main-first-with-n-to-3-frames.py:755`)，`main()` 中 `n_sample_frames=7` (`:830`)。
- 两段式模型切换 + tracker-only 优先 / detector 回退（与 2/3-frames 同构，`:596-674`）。

### 验证记录（2026-08-12 对话实测）
用数据 `[1945, 539, 532, 541, 540, 531, 548]`（7 帧）验证 `filter_n_to_best_3`：
- Q1=535.5, Q3=544.5, IQR=9.0，正常区间 `[522.0, 558.0]`；
- 帧 0 的 Y=1945 超上限 1387px，被判为异常并剔除（剩余 6 帧全部保留）；
- 6 > 3，择优取 位置 0/3/5 → 最终选中帧 `[1, 4, 6]`（对应原始帧号 12/49/74）；
- 结论：单一极端误检被干净剔除，正常帧无伤。

### 版本迭代 / 修复记录
| 日期 | 内容 |
|------|------|
| 2026-08-11 | 由 2/3-frames 派生：抽取逻辑改为等间距 n 帧 |
| 2026-08-12 | 加入 `filter_n_to_best_3`：IQR 剔异常 + 兜底补回 + 时序择优；`main()` 配置 n=7；实测验证筛选正确性 |

### 使用说明
路径在 `main()` 的 `paths`（`main-first-with-n-to-3-frames.py:811-816`）；抽帧数 `n_sample_frames` 在 `batch_process_videos_auto` 调用处设置（`:830`，当前 7）。运行即全自动：抽帧 → 识别 → 筛选 → tracker-only（回退 detector）→ 导出。

---

## 十、开发经验与教训总结

1. **显存是视频分割的最大敌人**：历经 `del + empty_cache` → `model.cpu()` → `_ALL_INFERENCE_STATES.clear()` → **子进程 spawn 隔离**（OS 级回收）四级演进，才解决批量长跑显存累积。
2. **以图引频策略有效**：先用轻量图像模型生成高质量 Prompt，比直接 `add_prompts` 手动框更稳；tracker-only + detector 双流互为兜底。
3. **统计筛选防单帧误检**：固定关键帧易被异常帧污染，n 帧 + IQR 剔异常 + 时序择优是最终的稳健方案。
4. **命名需与实现一致**：`sam3_video_double_point.py` 实际是框提示实现，属拷贝遗留，建议后续修正命名或实现。
