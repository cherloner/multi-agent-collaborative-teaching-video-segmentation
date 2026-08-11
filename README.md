# 多智能体协同驱动的教学视频智能切分平台

一套可直接运行的智能视频切片原型。前端使用原生 HTML、CSS 和 JavaScript；本地服务使用 FastAPI 与 FFmpeg，能够完成真实的视频上传、媒体探测、字幕语义与音画边界分析、视觉指纹去重、候选片段评分和 MP4 成片。

## 运行方式

确认本机已经安装 Python 3.10+ 与 FFmpeg，然后安装服务依赖并启动：

```powershell
python -m pip install -r requirements.txt
python server.py
```

然后访问 `http://localhost:4173`。

如果 `4173` 已被其他本地服务占用，可改用其他端口：

```powershell
$env:CUTPOINT_PORT = "4174"
python server.py
```

仅查看静态页面时仍可直接打开 `index.html`，但视频分析按钮需要通过 `server.py` 访问。

## 本地分析链路

1. 浏览器把视频发送到本机 FastAPI 服务。
2. `ffprobe` 读取时长、分辨率、帧率与音视频编码。
3. FFmpeg 检测场景切换、静音边界和黑帧区间。
4. 整条视频以 2 秒间隔单次解码为 `12×12 RGB` 视觉指纹，所有候选窗口复用这些样本计算内容相似度。
5. 时间轴融合器根据文本语义、边界证据、有效声音、有效画面和目标时长生成最多 3 条候选片段，并过滤画面高度重复的结果。
6. FFmpeg 输出 H.264/AAC MP4，浏览器展示预览和下载入口。
7. 固定数量的 worker 从 SQLite 持久队列领取任务，服务重启后会恢复仍有源文件的中断任务。
8. 分析结束后自动删除原始上传文件；生成短片保存在 `runtime/clips/`，该目录不会进入 Git。
9. 用户可对候选片段标记“保留/不合适”；系统冻结当时的七维评分特征，并在样本满足安全门槛后自动校准后续任务的排序权重。
10. 每个分析阶段与每条成片都会记录耗时，结果返回端到端处理时间和相对视频时长的实时倍率。
11. 服务启动时只在 `runtime/uploads/` 与 `runtime/clips/` 内执行引用感知的过期任务和孤儿文件清理。
12. 上传时单次流式计算 SHA-256；相同内容、目标时长、画幅、字幕开关、排序模型和处理签名完全一致时，复用已完成结果并复制为任务独立成片。

## 任务与反馈 API

- `POST /api/jobs`：上传视频并创建持久任务；表单支持 `target_duration`、`output_format` 与 `captions_enabled`
- `GET /api/jobs/{id}`：读取进度、结果和已有反馈
- `GET /api/jobs`：读取最近任务
- `POST /api/jobs/{id}/cancel`：取消排队或处理中任务；活动 FFmpeg/FFprobe 进程会被立即终止
- `POST /api/jobs/{id}/feedback`：写入或更新候选片段反馈
- `GET /api/feedback/stats`：汇总保留率
- `GET /api/feedback/model`：查看当前激活权重与最近一次校准结果
- `POST /api/feedback/calibrate`：使用现有反馈强制重跑离线校准
- `DELETE /api/jobs/{id}`：删除终态任务、成片和关联反馈

可通过环境变量调整本地资源边界：

```powershell
$env:CUTPOINT_WORKERS = "2"            # 1–4 个固定 worker
$env:CUTPOINT_MAX_PENDING_JOBS = "8"  # 排队与处理中任务总上限
$env:CUTPOINT_MAX_UPLOAD_MB = "2048"
$env:CUTPOINT_DB_PATH = "runtime/cutpoint.db"
$env:CUTPOINT_JOB_TIMEOUT_SECONDS = "3600"
$env:CUTPOINT_RUNTIME_MAX_GB = "20"
$env:CUTPOINT_MIN_FREE_GB = "1"
$env:CUTPOINT_RETENTION_DAYS = "30"
$env:CUTPOINT_ORPHAN_GRACE_HOURS = "24"
```

`CUTPOINT_RETENTION_DAYS=-1` 可关闭终态任务自动过期，`CUTPOINT_ORPHAN_GRACE_HOURS=-1` 可关闭孤儿文件清理。上传过程中若运行目录配额或磁盘保留空间不足，API 会返回 HTTP 507 并删除未完成文件。

FFmpeg、FFprobe、字幕提取、视觉采样和音频质检统一经过可中止进程执行器；处理中的取消请求与任务时长预算每 100ms 检查一次，触发后终止整个外部进程树并清理源文件和半成片。可选 ASR、OCR 或说话人模型若在 Python 进程内推理，则仍在模型调用前后执行协作式检查。

服务收到停止信号时使用独立的“可恢复中断”语义：终止当前外部进程、删除该任务的半成片、保留上传源并把状态退回 `queued`。下次启动会由 SQLite 恢复队列继续处理，不会把正常关机误记为用户取消或任务失败。

`GET /api/health` 返回当前运行目录用量、剩余配额、磁盘可用空间、启动维护结果、最近任务的中位/P95 耗时与中位实时倍率。单个任务结果的 `performance` 包含各分析阶段耗时、逐片渲染耗时、端到端耗时和 `source_seconds_per_processing_second`。

重复任务缓存不会共享同一个可删除文件：命中后仍会为新任务复制独立文件名，因此删除任一任务不会破坏另一个任务。处理签名由核心分析/渲染代码、依赖清单、FFmpeg 文件信息以及非敏感模型配置生成；代码、依赖、模型或 FFmpeg 变化后缓存自动失效。访问令牌不会进入签名、数据库或 API。

## 多画幅自动成片

创建任务时可选择以下 `output_format`：

- `source`：保持原画幅和接近原始分辨率；
- `landscape`：输出 `1280×720`（16:9）；
- `portrait`：输出 `720×1280`（9:16）；
- `square`：输出 `1080×1080`（1:1）。

固定画幅不会直接裁掉主体。渲染器会把原画面完整缩放到前景，并用同一画面的放大模糊版本填满剩余画布。所有输出统一使用 H.264、AAC、`yuv420p` 和 fast-start MP4，便于网页预览及平台上传。

`captions_enabled=true` 时，内嵌字幕或 ASR 文本会转换为片段相对时间的 ASS 字幕并烧录；OCR 角标不会被误当作对白。字幕针对不同画幅设置字号、换行和底部安全区，其中竖屏底部预留 220px。若当前 FFmpeg 缺少 libass 或字幕滤镜执行失败，系统会自动重试无字幕成片，并把降级原因写入 `export_profile.warnings` 和片段质量提示。

## 音频成片质量

存在音轨时，渲染器会在同一 FFmpeg 滤镜图中完成：

- EBU R128 单遍响度标准化，目标为 `-16 LUFS`、LRA 11、True Peak 上限 `-1.5 dBTP`；
- 重采样为 48kHz，并用异步重采样修正细小时间戳漂移；
- 片头 80ms、片尾 120ms 线性淡化，避免硬切产生爆音；
- AAC 160kbps 编码。

每条 MP4 写入磁盘后都会立即由 `ffprobe` 检查视频尺寸、目标时长、音轨存在性、音视频起始时间差和音视频时长差。任何关键检查失败都会删除不合格成片并让任务明确失败，不会把错误文件交付给前端。无音轨视频走独立的无声成片分支并接受对应质量检查。

## 反馈驱动的排序校准

每个候选片段都会保存 `speech_ratio`、`clean_frame_ratio`、`boundary_alignment`、`scene_density`、`semantic_score`、`visual_uniqueness` 和 `speaker_boundary_score` 七项归一化特征。视觉唯一性会在贪心多样性选择的每一轮动态重算，因此反馈提高该权重后，系统能在未达到硬去重阈值的候选之间主动偏好更新鲜的内容；说话人信号不可用时则从归一化权重中移除。

提交反馈时，实际可用的特征集合会随赞踩快照写入 SQLite，避免算法升级后历史样本失去可复现性。旧版五维样本仍可训练：校准器只在每条样本声明的可用特征内重新归一化，不会把缺失的视觉或说话人信号错误当作零分。

校准器使用确定性的受约束权重网格优化“保留片段应排在淘汰片段之前”的成对排序准确率，并设置以下保护：

- 至少 12 条可训练反馈；
- “保留”和“不合适”各至少 4 条；
- 样本至少来自 3 个不同任务；
- 每项信号权重不低于 5%、不高于 60%；
- 新权重只有在排序准确率或平均间隔实质提升时才激活；
- 样本不足、特征缺失或训练未提升时继续使用当前安全权重。

每个分析结果包含实际使用的 `scoring_profile` 版本。`GET /api/health` 也会返回激活模型与最近一次训练尝试，便于排查部署状态。

## 视觉指纹与候选去重

视觉采样完全复用系统已有的 FFmpeg，不需要 OpenCV 或额外视觉模型。每 2 秒提取一帧，经 area 缩放得到 `12×12 RGB` 描述子。帧级比较融合归一化亮度结构、边缘结构、强度分布和原始颜色，并在中心裁切尺度与 ±1 像素对齐范围内取最佳匹配，因此能够识别轻度裁切、亮度变化和调色后的近重复画面；低纹理纯色帧会回退到原始 RGB 距离，避免把不同标题卡误判为同一内容。

候选窗口之间继续使用双向最近邻画面相似度，默认相似度达到 `0.94` 时视为重复内容。确定性评测同时要求“中心裁切 + 三通道调色”后的重复相似度不低于 `0.94`，且与无关画面的相似度间隔不低于 `0.50`。

选择器先执行原有时间重叠约束，再执行视觉去重。结果中的 `signals.selection` 会记录候选总数、时间过滤数、视觉过滤数和实际阈值；每个入选片段还会包含 `visual.max_similarity` 与 `quality.visual_uniqueness`。指纹提取失败时会写入 `visual_warnings` 并自动回退到原有时间去重逻辑，不会阻断分析或成片。

## 可选语义提供器

嵌入字幕默认启用，不需要额外依赖。检测到字幕轨道后，系统会提取时间文本、把句子起止加入候选边界，并将文本覆盖率、信息密度和语句完整性纳入评分。

需要对无字幕视频做本地 ASR 或 OCR 时，可安装可选依赖：

```powershell
python -m pip install -r requirements-semantic.txt

$env:CUTPOINT_ASR_PROVIDER = "faster-whisper"
$env:CUTPOINT_WHISPER_MODEL = "small"
$env:CUTPOINT_WHISPER_DEVICE = "auto"
$env:CUTPOINT_WHISPER_COMPUTE_TYPE = "int8"

$env:CUTPOINT_OCR_PROVIDER = "tesseract"
$env:CUTPOINT_OCR_INTERVAL_SECONDS = "5"
$env:CUTPOINT_TESSERACT_LANG = "chi_sim+eng"
```

OCR 还需要系统安装 `tesseract` 可执行文件和对应语言包。所有可选提供器均为懒加载；依赖缺失或识别失败会记录在 `semantic_warnings`，不会中断场景、声音、黑帧和成片基线。

当前评分器已经融合可解释的时间文本、感知视觉唯一性、可选说话人轮次和真实反馈校准。后续路线是增加可选视觉语义嵌入与跨视频近重复索引。

## 可选说话人轮次检测

默认 `CUTPOINT_DIARIZATION_PROVIDER=off`，不加载大型音频模型。需要对访谈、播客或多人对话保护说话人轮次时，可安装：

```powershell
python -m pip install -r requirements-speaker.txt

$env:CUTPOINT_DIARIZATION_PROVIDER = "pyannote"
$env:CUTPOINT_HF_TOKEN = "你的 Hugging Face 访问令牌"
$env:CUTPOINT_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
$env:CUTPOINT_DIARIZATION_DEVICE = "auto"
```

所选 Pyannote 模型可能要求先在 Hugging Face 接受模型使用条件。令牌只从环境变量读取，不会写入任务数据库、日志或 API 响应。

轮次检测成功后，系统会：

- 把每个说话人轮次的起止点加入统一候选边界；
- 判断片段起止点是否落在持续发言中间；
- 将说话人边界保护分数融合进原有 `boundary_alignment`；
- 在候选证据和 `signals.speaker_timeline` 中返回说话人数、轮次数及匿名说话人标签。

Pyannote 未安装、令牌缺失、模型无权限或推理失败时，状态会出现在 `/api/health`、`speaker_providers` 和 `speaker_warnings`，任务仍会使用语义、场景、声音、画面与视觉去重完成分析。

## 测试

```powershell
python -m unittest discover -s tests -v
node --check script.js
python cutpoint_evaluation.py --strict
```

`cutpoint_evaluation.py` 会按 `evaluation/manifest.json` 自动生成带字幕轨道的确定性评测视频，验证场景边界、静音区间、黑帧区间、语义时间轴、高光召回、片段 IoU 和成片质量。完整报告写入 `runtime/evaluation/report.json`。

测试套件还会真实渲染 16:9、9:16 和 1:1 三种 MP4，使用 `ffprobe` 检查尺寸，并验证字幕时间裁剪、安全区、临时 ASS 清理与字幕失败降级。

当前质量门槛：

- 场景边界 F1 ≥ 0.95
- 静音与黑帧时间轴 IoU ≥ 0.95
- 语义时间轴 IoU ≥ 0.95
- 预期高光召回 = 1.0
- 平均高光 IoU ≥ 0.55
- 有效声音占比 ≥ 0.80
- 有效画面占比 ≥ 0.95
- 候选片段语义覆盖率 ≥ 0.60
- 候选片段视觉唯一性 ≥ 0.50
- 反馈校准后的成对排序准确率 ≥ 0.95
- 相对默认权重的成对排序准确率增益 ≥ 0.50
- 成片交付检查通过率 = 1.0
- 成片时长准确度 ≥ 0.80
- 音画同步准确度 ≥ 0.75
- 响度目标准确度 ≥ 0.80
- 首尾淡化有效度 ≥ 0.75
- 说话人边界保护分数 = 1.0
- 预期轮次切点准确度 = 1.0
- 分析实时倍率 ≥ 1.0
- 裁切与调色后的感知重复相似度 ≥ 0.94
- 感知重复与无关画面的相似度间隔 ≥ 0.50

## 动画逻辑

主视觉使用 8.4 秒循环，对应四个连续阶段：

1. 读取长视频，保留完整宽幅画面与音频波形。
2. 扫描多模态时间轴，扫描束从左向右移动。
3. 标记高价值切点，切点依次建立并展示语义、场景、情绪、静音与边界证据。
4. 生成短片，源画面退至背景，五条 15-60 秒演示切片从时间轴中分裂并展开。

高频动画只改变 `transform` 与 `opacity`。状态文字由低频 `setTimeout` 同步，不参与逐帧动画，因此不会阻塞其他交互。系统开启“减少动态效果”时，页面自动停在最终切片状态。

## 设计决策

- 视觉方向：以深色专业剪辑台为基底，使用单一冷蓝强调色与少量同色系紫光，避免高饱和霓虹。
- 信息结构：首屏采用不对称左右结构，右侧动画承担主要叙事，左侧只保留价值主张。
- 功能布局：四项能力使用 7:5 的非对称 Bento 网格，避免三张等宽卡片的模板感。
- 产品可信度：加入语义、场景、情绪、静音和边界证据，让“智能切片”从结果动画延伸到可解释过程。
- 交互入口：底部固定“开始试用”，呼吸环仅使用缩放与透明度；点击后打开可拖放视频的演示上传对话框。

## 响应式断点

- `1180px`：压缩首屏比例与发布格式视觉。
- `960px`：首屏改为上下布局，导航简化，功能网格调整为 6:6。
- `767px`：针对 375x812 设计为单列，主视觉缩短到 372px，仅展示三条切片，功能区全部纵向排列，固定 CTA 变为底部通栏。

## 素材

两张项目视觉素材由内置 ImageGen 生成，已转为 WebP：

- `assets/cutpoint-filmstrip.webp`
- `assets/cutpoint-studio.webp`

最终生成提示词方向为：电影化深色科技广告视觉、连续五段长视频画面、专业剪辑工作台、无文字、无标识、无水印。
