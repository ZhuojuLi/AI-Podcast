# AI 播客生成榜 —— 开发计划（榜单工程视角）

> 参考契约：`reference/starting-kit`（榜单官方契约参考，需自行获取）
> 目标：SUT 镜像实现 `/ready`、`/generate_audio`、`/generate_content`、`/download`，
> 在 A100 40G + 4C8G 环境稳定跑出可被人工评分的双人中文商业播客。

---

## 0. 榜单怎么赢（先对齐目标）

| 层级 | 内容 | 权重 | 策略 |
|---|---|---|---|
| Hard Gate | 中文 / 5–15min / 双人 / 性别正确 / 音频+文稿+封面齐全 / JSON 合法 | 一票否决 | 优先级最高，先做硬门禁 |
| 主分 | 文稿内容质量 40 + 对话质量 30 | 70 | 把火力集中在文字层 |
| 加分 | TTS 20 + 封面 10 | 30 | 稳中求好，不为封面烧大模型 |

核心判断：**这是个 LLM 可控的文字层占 70 分的工程题，不是 TTS 军备竞赛。**

目标观感不是「两个 TTS 轮流念商业分析」，而是「两个像真人的人在聊一个可信的商业故事」。

---

## 1. 基础镜像决策

**主选**：任意 vLLM 0.23 + cu129 基座镜像（Python3.12 / torch2.11 / vLLM0.23 / 自带 ffmpeg+curl）

实测（`docker run --rm` 验证）：
- Python 3.12.13 / torch 2.11.0+cu129 / vLLM 0.23.0（flashinfer 已内置）
- 已自带 `ffmpeg`、`curl`（curl 满足探活镜像要求）
- 缺：`flask gunicorn transformers soundfile librosa` → 由本工程 Dockerfile 补装
- `ENTRYPOINT` 为 `vllm serve`，**必须** `ENTRYPOINT ["/app/start.sh"]` 覆盖

**备选**：docker.io 上的 `vllm/vllm-openai` 对应版本（无预装，全量装依赖）
**非选**：`python:3.9`（starting-kit 用的）—— 无 CUDA/vLLM，不适合真实生成。

---

## 2. 本地可复用资产

| 环节 | 模型 | 位置 | 大小 |
|---|---|---|---|
| 文稿 LLM | Qwen3.8-27B-AWQ-INT4 | `models/Qwen3.8-27B-AWQ-INT4` | 20G |
| TTS 主 | Qwen3-TTS-12Hz-1.7B-CustomVoice + Tokenizer-12Hz | `comic2video/third_party/modelscope/models/Qwen/` | 4.3G + 651M |
| TTS 备 | Qwen3-TTS-12Hz-0.6B-CustomVoice | 同上 | — |
| ASR 回听 | Qwen3-ASR-0.6B + ForcedAligner-0.6B | 同上 | 1.8G |
| 安全审查 | Qwen3Guard-Gen-4B | 同上 | — |
| 封面图 | sdxl-turbo + sdxl_lightning_4step_unet；备 FLUX.2-klein-4B | `ceph/lizhuoju/image-model/`、`ceph/lizhuoju/models/` | 5–24G |

参考音色（voice reference）：**本地尚无，需要制作**（4–6 条，≥2 男 2 女）。

---

## 3. 目标架构（10 段流水线）

```
topic + speaker_gender1 + speaker_gender2
        │
        ▼
① Topic Resolver      实体/时间/事件消歧
        ▼
② Fact Retriever      SQLite FTS5 本地事实库 + LLM 已有知识
        ▼
③ Fact Ledger         原子事实/数字/日期/置信度（数字字段单独结构化）
        ▼
④ Story Planner       Hook→背景→冲突→转折→商业逻辑→影响→总结
        ▼
⑤ Dialogue Writer     双角色 persona：Story Host + Analyst
        ▼
⑥ Script QC           事实/中文/双人/时长预测/违禁表达/数字一致性
        ▼
   ┌────┴─────┐
   ▼          ▼
⑦ Dual TTS   Qwen3-TTS，turn-level 合成    ⑧ Cover（SDXL-Turbo + PIL 程序排版）
   ▼
⑨ Audio QC    Qwen3-ASR 回听 / CER / 时长 / 音量 / 静音
   ▼
⑩ Final Packager  mp3 + png + transcript.json
```

**关键设计约束**
- 文稿不能一次生成：`Fact Ledger → Outline → Dialogue` 拆开，缺数字时降级为描述性表达，不编造。
- TTS 按 turn（20–120 字）合成，失败只重生成单句；emotion prompt 与 spoken text 分离。
- 性别 → 音色写死映射（男/女各 ≥2 音色，避免同性别音色雷同）。
- 时长两层控制：TTS 前预估 6.5–10min，TTS 后 ffprobe 实测，越界触发扩写/删可选段。
- 封面文字用 PIL 绘制，图像模型只出无文字背景。

---

## 4. 显存预算（A100 40G，串行卸载）

| 阶段 | 常驻 | 峰值 |
|---|---|---|
| LLM 写稿 | Qwen3.8-27B-AWQ（vLLM, util~0.45）≈18–20G | |
| TTS | Qwen3-TTS-1.7B ≈4–5G | 与 LLM 可共存 |
| ASR | Qwen3-ASR-0.6B ≈2G | 与 LLM 可共存 |
| 封面 | SDXL-Turbo lazy-load ≈7G | 释放 LLM KV 后再加载 |

策略：LLM + TTS + ASR 常驻，封面模型最后 lazy-load（不影响首字延迟）。

---

## 5. 提交形态

- 代码 + 运行时依赖打进镜像；模型权重**不进镜像**。
- 运行时经 Ceph 挂载：`mountPoint: /models` + `srcRelativePath: lizhuoju/<dir>`。
- `start.sh`：后台起 vLLM(8100) → 轮询 `/v1/models` 就绪 → 前台起主服务(80)。
- `submit.yaml` 声明 `inferenceImage` / `inferenceVolumeMounts` / `env`。

---

## 6. 工程骨架（已落地）

```
app/
├── main.py              # 契约层：4 个接口（已实现）
├── config.py            # 环境变量配置（已实现）
├── schemas.py           # Transcript/Turn（已实现）
├── state.py             # item 会话 + 文稿队列（已实现）
├── pipeline.py          # 生成流水线编排（已实现）
└── providers/
    ├── base.py          # ScriptProvider / TTSProvider / CoverProvider 抽象（已实现）
    ├── mock.py          # 默认 mock：ffmpeg 静音 + Pillow 封面（已实现，过机器校验）
    ├── registry.py      # 可插拔加载（已实现）
    ├── llm.py           # TODO：Qwen3.8 写稿
    ├── tts.py           # TODO：Qwen3-TTS turn-level
    ├── asr.py           # TODO：Qwen3-ASR 回听
    └── image.py         # TODO：SDXL-Turbo 封面
config/
├── runtime.yaml         # 阈值/开关
├── voices.yaml          # 音色库 + 性别映射
└── prompts.yaml         # Fact/Outline/Dialogue/QC 提示词
qc/                      # TODO：硬门禁
resources/               # 事实库 / 参考音 / 封面模板
scripts/
└── smoke_test.py        # 复刻评测端校验（已实现）
```

接入真实后端**无需改契约层**，只改 `SCRIPT_PROVIDER / TTS_PROVIDER / COVER_PROVIDER` 三个环境变量。

---

## 7. Hard Gate 清单（任一 FAIL 不出结果）

| 检查 | 级别 |
|---|---|
| MP3 / PNG / JSON 存在且可解析 | HARD |
| 中文占比 > 90% | HARD |
| speaker 仅 1/2，双方轮次均 ≥ N | HARD |
| 性别与音色 metadata 匹配 | HARD |
| 实测时长 300–900s（目标 420–600s） | HARD |
| 音频非空 / 无长静音 | HARD |
| ASR CER < 8%（目标 < 3%） | HARD |
| 封面 PNG 1024×1024 | HARD |
| 数字/日期均来自 Fact Ledger | HARD |
| 轮次比例 40–60% / filler 频率 | SOFT |
| 对话互动度 / 音量 | SOFT |

---

## 8. 一周计划

| Day | 任务 | 验收 |
|---|---|---|
| **D1** | 定 TTS：Qwen3-TTS vs CosyVoice3 vs IndexTTS2.5 bake-off（10 组固定文本，测 MOS/CER/RTF/数字/英文/情绪） | 定 TTS 主方案 + 参考音色 |
| **D2** | 单条 E2E 跑通：topic→LLM→script→TTS→mp3→cover→json | 连续 10 case：100% 有输出/时长正确/性别正确 |
| **D3** | Story Pipeline：Fact Ledger→Outline→Dialogue，加 persona/section/fact_id | 只读 transcript 不觉得像 AI |
| **D4** | Fact Safety：claim/number/date 校验 + 本地 business_corpus.db(FTS5) | retrieval 命中，unsupported number≈0 |
| **D5** | Audio QC：ASR+CER+retry+duration+loudness+silence → 真 HardGate | 50 case：0 越界/0 空音频，P95 CER<5% |
| **D6** | 封面（SDXL-Turbo + PIL 排版）+ 性能（warmup/RTF/首字） | RTF<0.5，首字<20s |
| **D7** | 100 case 回归：topic 10 类 × 性别 4 种全覆盖 | 失败率<2%，全部 Hard Gate 通过 |

---

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| Qwen3-TTS 源码安装坑（PyPI 0.1.1 不支持 12Hz） | 源码安装，或从现有 `.venv-tts` 硬链克隆环境 |
| 40G 单卡装不下全部模型 | LLM+TTS+ASR 常驻，封面 lazy-load |
| 首字延迟 | 先出前 3–4 turn 即合成首段，边写边合成 |
| 事实幻觉 | Fact Ledger 硬约束；缺数据改描述性表达 |
| 同性别音色雷同 | 男/女各备 ≥2 音色，registration 前交叉检查 |
| Ceph 目录权限 | `sudo -n mkdir/chown`（见 skills/model-deploy/vllm-docker-deploy.md） |

---

## 10. 复刻评测端自测

```bash
python3 -m app.main          # 本地 mock，默认 80，可用 MOCK_SERVER_PORT 改
python3 scripts/smoke_test.py --base-url http://localhost:8086
```

`smoke_test.py` 复刻 `run_for_darvin.py` 的核心校验：并发双流、SSE 事件序列、
音频时长 300–900s、首 chunk <30s、封面 1024×1024 PNG。
