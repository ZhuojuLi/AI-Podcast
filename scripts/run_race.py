# -*- coding: utf-8 -*-
"""模型赛马执行器：题库 × provider → 音频 + 盲听评分页。

provider 约定（新增模型只需满足其一）：
- cosy     内置：容器内跑 CosyVoice3（走 app.providers.cosy）。
- 外部目录 <model-race>/<name>/race_synth.py：
    python race_synth.py --script <bank.json> --out <out.mp3>
  存在即自动加入赛马；不存在则跳过。

盲听 HTML：所有音频随机化名（A/B/C…），真实对应关系存 mapping.json，
评分项：Naturalness / Prosody / Turn Flow / Expressiveness / Listenability。

用法（CosyVoice 基线）：
  docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=3 -e \
    TTS_MODEL_PATH=/work/.models/Fun-CosyVoice3-0.5B-2512 --entrypoint bash \
    -v <ai-podcast仓库>:/work -v <model-race目录>:/mr \
    -w /work <构建出的镜像名> -c \
    "PYTHONPATH=/work:/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
     COSYVOICE_PRECOMPUTED_REFS=/work/resources/voices/reference_features.pt \
     /opt/podcast-venv/bin/python scripts/run_race.py --bank output/eval_bank \
     --providers cosy --out output/race_round1"
"""
import argparse
import html
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRICS = ["naturalness", "prosody", "turn_flow", "expressiveness", "listenability"]

# provider 短名 -> model-race 下的目录名
PROVIDER_DIRS = {
    "firered": "FireRedTTS2",
    "soulx": "SoulX-Podcast",
    "moss": "MOSS-TTSD",
}
# 固定盲听编号：跨题材/跨批次一致，支持多容器按 provider 分组并行跑
PROVIDER_CODES = {"cosy": "A", "firered": "B", "soulx": "C", "moss": "D"}
PROVIDER_NAMES = {v: k for k, v in PROVIDER_CODES.items()}


def synth_cosy(script_path: Path, out_mp3: Path) -> None:
    from app.providers.cosy import CosyTTSProvider
    from app.schemas import Delivery, Transcript, Turn

    obj = json.loads(script_path.read_text(encoding="utf-8"))
    transcript = Transcript(
        title=obj.get("title", ""),
        turns=[Turn(speaker=int(t["speaker"]), text=str(t["text"]),
                    delivery=Delivery.coerce(t["delivery"]) if t.get("delivery") else None)
               for t in obj.get("content", [])],
    )
    provider = CosyTTSProvider()
    chunks = [c for c in provider.stream_mp3(transcript, "男", "女")]
    out_mp3.write_bytes(b"".join(chunks))


def synth_external(race_root: Path, name: str, script_path: Path,
                   out_mp3: Path) -> None:
    runner = race_root / name / "race_synth.py"
    # 每个模型有自己的 venv（依赖与主环境隔离），优先使用
    venv_py = race_root / name / "venv" / "bin" / "python"
    exe = str(venv_py) if venv_py.exists() else sys.executable
    env = dict(os.environ)
    if name == "SoulX-Podcast":
        cache = race_root / "models" / "SoulX-Podcast" / "s3tokenizer_cache"
        if cache.is_dir():
            env["XDG_CACHE_HOME"] = str(cache)
    proc = subprocess.run(
        [exe, str(runner), "--script", str(script_path),
         "--out", str(out_mp3)],
        capture_output=True, text=True, timeout=3600, env=env)
    if proc.returncode or not out_mp3.exists():
        raise RuntimeError(f"{name} 合成失败: {proc.stderr[-800:]}")


def build_review_html(out: Path, rows, mapping) -> None:
    """rows: [(script_stem, [(code, mp3_path), ...])]"""
    items = []
    rnd = random.Random(20260923)
    for stem, entries in rows:
        codes = list(entries)
        rnd.shuffle(codes)
        cards = []
        for code, rel in codes:
            cards.append(f"""
      <div class="card">
        <div class="code">{code}</div>
        <audio controls src="{html.escape(rel)}"></audio>
        <table>""" + "".join(
                f'<tr><td>{m}</td>' +
                "".join(f'<td><label><input type="radio" name="{stem}-{code}-{m}" value="{v}">{v}</label></td>'
                        for v in range(1, 6))
                + "</tr>"
                for m in METRICS) + """
        </table>
      </div>""")
        items.append(f"<h2>{html.escape(stem)}</h2>\n" + "\n".join(cards))
    page = f"""<!doctype html><meta charset="utf-8">
<title>Podcast Race 盲听评分</title>
<style>
body{{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}}
.code{{font-size:22px;font-weight:700}}
td{{padding:2px 8px;font-size:14px}}
h1{{font-size:22px}}
</style>
<h1>Podcast Race 盲听评分（1-5 分，凭听感，不要猜是谁）</h1>
{''.join(items)}
"""
    (out / "review.html").write_text(page, encoding="utf-8")
    (out / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="题库目录（build_eval_bank.py 产物）")
    ap.add_argument("--providers", default="cosy", help="逗号分隔： cosy,firered,soulx,moss…")
    ap.add_argument("--out", required=True)
    ap.add_argument("--race-root", default="/mr")
    args = ap.parse_args()

    bank = Path(args.bank)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    race_root = Path(args.race_root)

    wanted = [p.strip() for p in args.providers.split(",") if p.strip()]
    scripts = sorted(bank.glob("*.json"))
    old_mapping = {}
    mapping_path = out / "mapping.json"
    if mapping_path.exists():
        old_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows, mapping = [], {}
    for script_path in scripts:
        stem = script_path.stem
        (out / stem).mkdir(exist_ok=True)
        # 固定字母映射（PROVIDER_CODES）：已存在的音频直接登记，缺哪个补哪个，
        # 支持按 provider 分组的多容器并行（cosy 在 A 容器、其余在 B 容器）。
        entries, code_map = [], {}
        existing = {p.stem: p for p in (out / stem).glob("*.mp3")}
        for code in sorted(existing):
            entries.append((code, f"{stem}/{existing[code].name}"))
            code_map[code] = (old_mapping.get(stem, {}).get(code)
                              or PROVIDER_NAMES.get(code, "?"))
        for name in wanted:
            code = PROVIDER_CODES.get(name)
            if code is None:
                code = chr(ord("A") + len(entries))
            mp3 = out / stem / f"{code}.mp3"
            if mp3.exists():
                print(f"[race] {stem} × {name} 已存在，跳过", flush=True)
                if code not in code_map:
                    entries.append((code, f"{stem}/{mp3.name}"))
                    code_map[code] = name
                continue
            if name != "cosy":
                directory = PROVIDER_DIRS.get(name, name)
                if not (race_root / directory / "race_synth.py").exists():
                    print(f"[race] 跳过 {name}："
                          f"{race_root / directory}/race_synth.py 不存在", flush=True)
                    continue
            t0 = time.time()
            try:
                if name == "cosy":
                    synth_cosy(script_path, mp3)
                else:
                    synth_external(race_root, PROVIDER_DIRS.get(name, name),
                                   script_path, mp3)
            except Exception as exc:
                print(f"[race] {stem} × {name} 失败: {exc}", flush=True)
                continue
            dur = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(mp3)], capture_output=True, text=True)
            print(f"[race] {stem} × {name} -> {mp3.name} "
                  f"{dur.stdout.strip()}s, {time.time() - t0:.0f}s", flush=True)
            entries.append((code, f"{stem}/{mp3.name}"))
            code_map[code] = name
        rows.append((stem, entries))
        mapping[stem] = code_map

    build_review_html(out, rows, mapping)
    print(f"[race] review.html + mapping.json -> {out}", flush=True)


if __name__ == "__main__":
    main()
