# -*- coding: utf-8 -*-
"""封面生成后端。

  - PillowCoverProvider（name="pil"）：程序化排版封面，无需 GPU，立即可用。
  - SDXLCoverProvider（name="sd"）：SDXL-Turbo 出无文字背景 + PIL 排版（待接入）。

标题由程序绘制（避免 diffusion 生成乱码），输出固定 1024x1024 PNG。
"""
import os
import random
import textwrap

from app.config import config
from app.providers.base import CoverProvider

CJK_FONT_CANDIDATES = [
    os.getenv("COVER_FONT", ""),
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]

PALETTES = [
    ((18, 32, 58), (54, 110, 168)),    # 深蓝
    ((32, 24, 18), (176, 108, 44)),    # 咖啡
    ((12, 38, 32), (36, 128, 96)),     # 墨绿
    ((40, 16, 30), (168, 58, 92)),     # 酒红
    ((24, 22, 40), (96, 84, 180)),     # 紫
]


def _load_font(size: int):
    from PIL import ImageFont

    for path in CJK_FONT_CANDIDATES:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_cjk(text: str, per_line: int):
    text = text.strip()
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=per_line) or [""])
    return lines


class PillowCoverProvider(CoverProvider):
    name = "pil"

    def __init__(self, width: int = None, height: int = None) -> None:
        self.width = width or config.COVER_REQUIRED_WIDTH
        self.height = height or config.COVER_REQUIRED_HEIGHT

    def generate(self, item_id: str, title: str, topic: str, out_path: str) -> bool:
        from PIL import Image, ImageDraw

        W, H = self.width, self.height
        top, bottom = random.choice(PALETTES)

        # 垂直渐变背景
        img = Image.new("RGB", (W, H))
        px = img.load()
        for y in range(H):
            t = y / (H - 1)
            px_row = (
                int(top[0] + (bottom[0] - top[0]) * t),
                int(top[1] + (bottom[1] - top[1]) * t),
                int(top[2] + (bottom[2] - top[2]) * t),
            )
            for x in range(W):
                px[x, y] = px_row
        draw = ImageDraw.Draw(img)

        # 顶部品牌
        brand_font = _load_font(40)
        draw.text((80, 80), "AI 播客", font=brand_font, fill=(255, 255, 255))

        # 分隔线
        draw.rectangle([80, 150, W - 80, 156], fill=(255, 255, 255))

        # 主标题（最多 3 行）
        title_lines = _wrap_cjk(title.replace("AI 播客：", ""), 9)[:3]
        title_font = _load_font(96)
        y = 300
        for line in title_lines:
            draw.text((80, y), line, font=title_font, fill=(255, 255, 255))
            y += 130

        # 副标题：主题
        sub_font = _load_font(44)
        sub_lines = _wrap_cjk(topic or "", 16)[:2]
        y += 40
        for line in sub_lines:
            draw.text((80, y), line, font=sub_font, fill=(235, 235, 235))
            y += 60

        # 底部
        foot_font = _load_font(36)
        draw.text((80, H - 120), "对话 · 商业故事 · 洞察", font=foot_font, fill=(220, 220, 220))

        img.save(out_path, "PNG")
        return True


class SDXLCoverProvider(CoverProvider):
    name = "sd"

    def __init__(self, model_path: str = "", device: str = "cuda:0") -> None:
        self.model_path = model_path or os.getenv("COVER_MODEL_PATH", "")
        self.device = device
        self.width = config.COVER_REQUIRED_WIDTH
        self.height = config.COVER_REQUIRED_HEIGHT

    def generate(self, item_id: str, title: str, topic: str, out_path: str) -> bool:
        raise NotImplementedError(
            "SDXLCoverProvider 待实现：SDXL-Turbo 生成无文字背景 + PIL 排版标题（D6）"
        )
