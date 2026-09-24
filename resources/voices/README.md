# 参考音色（需自备）

本目录**不包含**参考音频。CosyVoice3 的声音克隆需要 4 段参考 WAV：

| 文件 | 用途 |
|---|---|
| `male.wav` | 男声主播（异性组合的主音色） |
| `female.wav` | 女声主播（异性组合的主音色） |
| `male2.wav` | 第二男声（男男组合时 speaker2 使用） |
| `female2.wav` | 第二女声（女女组合时 speaker2 使用） |

要求：24kHz 单声道 WAV、9~12 秒、连续单人说话、无叠话无背景音乐、
响度饱满；文本内容与 `cosy.py` 中的参考转写一致（自行改写）。

选段工具链（见 `scripts/`）：
1. `scan_podcast_voices.py`：campplus 说话人嵌入全片扫描，输出候选窗与纯度/能量指标
2. F0（pyworld）标性别
3. `trim_reference_voice.py`：70Hz 高通 → 峰值归一 -3dB → 首尾 60ms 静音垫 + 20ms 淡入淡出
4. `precompute_cosy_refs.py`：预计算声学特征到 `reference_features.pt`（sha256 校验）

> ⚠️ 请只使用你拥有合法使用权的声音。公开发布克隆他人（尤其公众人物）
> 声音的参考音频或合成音频存在法律风险，本项目不提供任何参考音频。
