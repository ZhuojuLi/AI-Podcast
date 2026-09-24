# AI-Podcast

输入一个中文商业故事主题与两位主播的性别，输出一期完整的中文双人对话播客：
**文稿 + 音频 + 封面**，以 HTTP 服务形式提供。

- 文稿：Qwen3.8-27B 生成（vLLM 推理，本地开发可用 14B-AWQ），支持联网搜索辅助写稿
- 音频：FireRedTTS-2 双人对话合成（四候选盲听赛马胜出；镜像内置 CosyVoice3 为默认后端）
- 封面：PIL 中文封面（默认，零依赖）/ SD 后端可插拔

全部模型均为 Apache-2.0，可全程离线运行，**无需任何付费 API**。

## 快速开始（本地 mock，无需 GPU）

```bash
pip install -r requirements.txt
MOCK_SERVER_PORT=8086 python3 -m app.main
# 另开终端
python3 scripts/smoke_test.py --base-url http://localhost:8086
```

mock 后端用 ffmpeg 生成 7–10 分钟静音 MP3、Pillow 生成 1024×1024 封面、
固定话术文稿，用于先跑通接口契约与机器校验（非评分用）。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/ready` | 探活，LLM + TTS 均就绪后返回 `200 True`，此前 503 |
| POST | `/generate_audio` | `x-www-form-urlencoded` → chunked 原始 MP3 字节流 |
| POST | `/generate_content` | SSE 文稿流（`content_start`→`content`→`done`） |
| GET | `/status/<item_id>` | 会话生命周期：`processing` / `completed` / `failed`（含失败原因与耗时） |
| GET | `/download/<file_type>/<file_id>` | 下载 `image`/`audio` 文件 |

`/generate_audio` 参数：`item_id`（必填）、`topic`、`speaker_gender1`、`speaker_gender2`。

## 本地开发（真实模型）

```bash
# 1) 起写稿 LLM（vLLM）
LLM_MODEL_DIR=<本地量化模型目录> bash scripts/start_llm_dev.sh

# 2) 生成一条 demo（LLM 写稿 → 双人 TTS → 封面）
CUDA_VISIBLE_DEVICES=0 <venv>/bin/python scripts/make_demo.py \
    --topic "瑞幸如何靠生椰拿铁翻盘" --g1 男 --g2 女 --target-seconds 150

# 3) 起契约服务（真实 provider）
bash scripts/start_sut_dev.sh
curl -s http://127.0.0.1:8086/ready
```

产物在 `output/demo/`：`1_audio.mp3`、`1_cover.png`、`transcript.json`、
`index.html`（可直接打开试听）。

## ⚠️ 部署前必办

**参考音色音频没有随仓库发布**（克隆公众人物声音存在法律风险）。
CosyVoice3 默认后端需要 4 段你有合法使用权的参考 WAV（男/女 × 主/副音色），
缺失时合成出的音色与请求的性别/风格无关。必须：

1. 准备 4 段 24kHz 单声道 WAV 放入 `resources/voices/`
   （要求与选段工具链见 [resources/voices/README.md](resources/voices/README.md)）；
2. 运行 `python3 scripts/precompute_cosy_refs.py` 重算声学特征
   `reference_features.pt`（含文件名索引 + sha256 校验，防止特征与音频错位）；
3. **人工试听**：确认音色区分度、性别与 `speaker_gender1/2` 标注一致。

模型挂载点不同时改 `MODELS_ROOT` 环境变量（默认 `/models`）。

## 模型赛马（为什么最终是 FireRedTTS-2）

10 题材评测题库（知识讲解/强观点/轻松聊天/情绪故事/数字密集/快速问答/附和接话/
争论打断/长回答/幽默），同稿多模型合成 + 盲听评分页（Naturalness / Prosody /
Turn Flow / Expressiveness / Listenability 五项指标），基建见
`scripts/build_eval_bank.py` + `scripts/run_race.py`：

| 模型 | 听感结论 |
|---|---|
| **FireRedTTS-2** ✅ | 停顿与对话感最强，韵律自然；对参考音声学环境敏感（带 BGM 的参考会连背景一起克隆） |
| CosyVoice3-0.5B | 克隆保真度高、稳定，但 instruct2 统一指令会把不同音色抹成同一种风格 |
| SoulX-Podcast | 播客专项，对话结构好 |
| MOSS-TTSD | 韵律平淡、听感偏差，出局 |

FireRed 适配器见 `scripts/race/firered_synth.py`（整段对话模式，轮替/韵律由
模型内部决定）。当前提交镜像内置 CosyVoice3（参考音色克隆 + instruct 指令），
provider 层可插拔，切换只需实现 `app/providers/base.py` 的抽象接口。

## 音色工程

参考音不是"随便截一段"就能用的，踩过的坑都在工具链里：

- `scripts/auto_reference.py`：**全自动冷启动**——输入整集未裁剪播客，
  滑窗提取 campplus 嵌入 → 球面 k-means 聚类自动发现说话人（无需先验
  参考音）→ 按纯度/句间底噪/信噪比/能量动态打分挑段 → 修整输出
  `male.wav`/`female.wav`。原则是**筛选优先于清洗**：句间底噪会被克隆进
  合成结果，带 BGM/混响的段直接丢弃，候选不足时才用 `--denoise`
  （DeepFilterNet）事后补救
- `scripts/scan_podcast_voices.py`：已知目标音色时的精细复扫（9s 窗 /
  1.5s 跳距），按嵌入相似度、纯度、能量动态、响度打分输出候选清单
- F0（pyworld）标性别 + 嵌入相似度校验身份
- `scripts/trim_reference_voice.py`：70Hz 高通去低频隆隆声 → 峰值归一 -3dB →
  首尾 60ms 静音垫 + 20ms 淡入淡出（不修的话克隆出的语音首尾会带杂音/爆音）
- 双音色设计：评测的性别组合可能是男男/女女，单一音色必丢"区分度"分，
  v1.5 起 speaker2 自动切换第二套音色与配套人设指令

## 架构边界（已知限制）

- **会话生命周期**：`processing / completed / failed` 三态，
  `GET /status/<item_id>` 可查；合成异常、文稿异常、客户端中断都会落
  `failed` 并带原因，不会误标 `completed`（有回归测试）
- **单实例语义**：会话状态在进程内存、队列无界、TTS 推理用全局锁串行。
  这是单实例服务的实现方式；要水平扩展需外置状态存储与队列、按实例
  并发控制和监控配套，不在本仓库范围内

## 显存预算（A100，LLM + TTS 同卡）

| 项 | 显存 |
|---|---|
| Qwen3.8-27B / Qwen3-14B-AWQ + vLLM（显式预算 `LLM_MEM_GB`，默认 24G） | ~24 GB |
| CosyVoice3 fp16 + CUDA 运行时 | ~3–4 GB |
| **合计** | **~28 GB** |

显存按绝对预算自动换算：`start.sh` 读取实际 GPU 总显存，把 `LLM_MEM_GB`
换算成 `--gpu-memory-utilization`（40G 卡 → 0.60；80G 卡 → 0.30），余量给 TTS。

## 实测（v1.3，14B-AWQ + CosyVoice3，4 核 / 8GiB RAM）

延迟分三个口径报告，勿混用：

| 指标 | 含义 | v1.3 | v1.0（27B，A100-80G） |
|---|---|---|---|
| 首个响应字节 | HTTP 流第一个非空 chunk 到达 | 10.76s | 13.72s |
| 首段可播放语音 | 跳过前导 ID3 标签后，首个 MP3 音频帧到达 | = 10.76s | = 13.72s |
| 整单 / RTF | 全单完成时间 / 实时率 | 276.8s / 0.660 | 290.8s / 0.760 |

- 成品：69 轮对话、419.4 秒音频（v1.0：55 轮、382.8 秒），
  音频/文稿 SSE/1024² 封面均通过 `scripts/smoke_test.py`
- 整单结束后容器无 OOM，稳态内存约 5.7GiB / 8GiB
- v1.0：MP3 16kHz 单声道，下载接口与流式内容 SHA-256 一致

**口径说明**：上表两个版本测量时 `TTS_EARLY_ID3_KIB=0`（未提前发送），
所以"首个响应字节"就是"首段可播放语音到达"。仓库 Dockerfile 默认
`TTS_EARLY_ID3_KIB=1024`（评测 30 秒首 chunk 门禁的保底措施，见踩坑速查），
开启后"首个响应字节"测到的是 1 MiB ID3 元数据、亚秒级即达，
**不再代表用户何时听到第一句话**。`smoke_test.py` 现在分别输出
`first_byte` 与 `first_audio`：后者解析 ID3v2 syncsafe 长度跳过元数据、
定位第一个 MP3 帧同步头（`0xFFEx`），才是用户感知口径。

## 四个 0 分门限（防御措施）

| 门限 | 防御 |
|---|---|
| 必须中文 | Prompt 硬约束 + CJK 占比检测（整篇 0.85，逐幕 0.5——数字密集幕不套用整篇阈值） |
| 5–15 分钟 | 三重保险：字数反算 + max_tokens 硬限 + ffprobe 实测校验（越界按偏差缩放字数目标重生成） |
| 双人对话 | `[S1]`/`[S2]` 结构强制 + 轮次均衡校验（各占 ≥25%） |
| 性别匹配 | 按性别选音色（构造保证）+ 配置层校验；音色性别由参考音频保证 |

## 踩坑速查

- **Qwen3.5 是混合 Mamba 架构**：vLLM 下不能开 `--enable-prefix-caching`
  （会把 `mamba_cache_mode` 设为 `all` 并抛 `NotImplementedError`，engine 直接挂）
- **首字节 30 秒门禁**：改为按段流式——首段只写 2 行短句立刻合成出声，
  其余边写边合成，首字节 ~35s → 10.7s
- **续写解析**：模型续写用 `3|4|5|…` 连续编号，解析器只认 1/2 会整段丢失、
  文稿过短；改为任意序号按奇偶映射
- **TTS 预热要真跑一句**：只加载权重的话首个 case 要触发 kernel 调优，
  第一句 35~44s；预热真合成后稳态首字节 16~21s
- **音频流首 chunk 保底**：流开头立即输出 1 MiB 合法 ID3v2.3 空白元数据
  （解码器跳过，不影响时长与听感），防止下游读取缓冲把首 chunk 拖过门禁。
  代价：此后"首个非空 chunk"不再代表可播放语音到达，测延迟必须用
  `smoke_test.py` 的 `first_audio` 口径
- **FireRed 参考音要带干声**：播客片段带房间混响会被连同克隆；合成干声
  prompt 或选无混响片段能消除回响

## 容器运行

模型不烘进镜像，运行时挂载：

```bash
docker build -t ai-podcast .
docker run --gpus all -it --rm \
    -v /host/models:/models \
    -v /host/out:/out \
    -e MODELS_ROOT=/models \
    ai-podcast bash
```

镜像内：`start.sh` 后台起 vLLM(8100) → 探活 `/v1/models` → 前台契约服务(80)，
`/ready` 在两者都就绪后返回 200。

## 可插拔生成层

契约层不关心生成实现，通过环境变量切换：

| 变量 | 取值 | 实现 |
|---|---|---|
| `SCRIPT_PROVIDER` | `mock` / `qwen` / `module:Class` | 文稿 |
| `TTS_PROVIDER` | `mock` / `qwen` / `cosy` / `module:Class` | 音频 |
| `COVER_PROVIDER` | `mock` / `sd` / `pil` / `module:Class` | 封面 |
| `SEARCH_PROVIDER` | `ddgs`（免 key DuckDuckGo 聚合）/ `none` | 写稿前联网搜索 |

`SEARCH_PROVIDER=ddgs` 时，写稿前按主题联网搜索、抓正文裁成 ≤1800 字摘要
注入 prompt（要求"资料中没有的具体数字不要编造"），搜索与写稿后台并发
（首段只等 6s 保首字节），失败自动降级为无搜索行为。

## 目录结构

```
app/                 契约服务层（main 接口 / pipeline 编排 / state 会话）
app/providers/       可插拔生成层（llm 文稿 / cosy TTS / image 封面 / search 搜索）
config/              prompt、运行时阈值、音色与 persona 配置
scripts/             冒烟 / demo / 赛马 / 音色选段与预计算 / 开发起停脚本
scripts/race/        FireRedTTS-2 赛马适配器
tests/               单元测试
third_party/CosyVoice  CosyVoice 源码（Apache-2.0，镜像构建时 vendored 进去）
resources/voices/    参考音色（需自备，见该目录 README）
docs/PLAN.md         设计与实现计划
```

## 许可与致谢

本项目代码以 Apache-2.0 发布（见 LICENSE）。

依赖的模型全部为 Apache-2.0：

- Qwen3.8-27B / Qwen3-14B-AWQ — 文稿生成
- CosyVoice3（FunAudioLLM/CosyVoice）— 参考音色克隆 TTS（镜像默认后端）
- FireRedTTS-2 — 双人对话 TTS（赛马优胜模型，适配器随仓）

**参考音色音频不包含在本仓库中**。克隆他人（尤其公众人物）声音需要获得
合法授权；请自备你拥有使用权的参考音频，并在公开使用合成音频前确认合规。
