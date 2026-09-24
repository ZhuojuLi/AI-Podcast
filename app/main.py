# -*- coding: utf-8 -*-
"""AI 播客生成 SUT 契约层（Flask）。

接口：
  GET  /ready                             探活（返回 200 "True"）
  POST /generate_audio                    chunked 原始 MP3 字节流
  POST /generate_content                  SSE 文稿流（content_start → content → done）
  GET  /download/<file_type>/<file_id>    下载音频 / 封面图

真实生成能力由 SCRIPT_PROVIDER / TTS_PROVIDER / COVER_PROVIDER 决定，
详见 app/providers/。
"""
import json
import os
import sys
import threading
import time
from typing import Any, Dict

import requests
from flask import Flask, Response, request, send_file

from app.config import config
from app.pipeline import audio_stream, content_events, start_session
from app.state import STORE


def json_response(data: Dict[str, Any], status_code: int = 200) -> Response:
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        status=status_code,
        mimetype="application/json; charset=utf-8",
    )


def _stream_headers(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"
    return resp


def _param(name: str, default: str = "") -> str:
    """同时兼容 form-urlencoded 与 JSON 请求体。"""
    if name in request.form:
        return request.form.get(name, default) or default
    payload = request.get_json(silent=True)
    if isinstance(payload, dict) and name in payload:
        value = payload.get(name)
        return default if value is None else str(value)
    return default


_LLM_READY_CACHE = {"ts": 0.0, "ok": False}
_TTS_READY = threading.Event()


def _llm_ready() -> bool:
    """LLM（vLLM）是否就绪；结果缓存 3 秒，避免探活打爆后端。"""
    now = time.time()
    if now - _LLM_READY_CACHE["ts"] < 3.0:
        return _LLM_READY_CACHE["ok"]
    ok = False
    try:
        resp = requests.get(f"{config.LLM_BASE_URL.rstrip('/')}/models", timeout=3)
        ok = resp.status_code == 200
    except Exception:
        ok = False
    _LLM_READY_CACHE.update(ts=now, ok=ok)
    return ok


def _preload_tts() -> None:
    try:
        # 先等 LLM 就绪再加载 TTS，避免与 vLLM 初始化争抢显存
        if config.READY_REQUIRE_LLM and config.SCRIPT_PROVIDER == "qwen":
            deadline = time.time() + 3600
            while time.time() < deadline and not _llm_ready():
                time.sleep(5)
        from app.providers.registry import get_tts_provider

        provider = get_tts_provider()
        preload = getattr(provider, "preload", None)
        if callable(preload):
            print("[startup] 预热 TTS 模型...", flush=True)
            preload()
            print("[startup] TTS 预热完成", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] TTS 预热失败，/ready 保持 503: {exc}", flush=True)
    else:
        _TTS_READY.set()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    @app.route("/ready", methods=["GET"])
    def ready():
        # 需要 LLM 时，vLLM 未就绪返回 503，评测端 readinessProbe 会持续重试
        if config.READY_REQUIRE_LLM and config.SCRIPT_PROVIDER == "qwen" and not _llm_ready():
            return Response("LLM not ready", status=503, mimetype="text/plain")
        # 需要 TTS 时，等预热完成再返回 200，避免首个 case 首字节被冷加载拖慢
        if (config.READY_REQUIRE_TTS and config.TTS_PRELOAD
                and config.TTS_PROVIDER in ("qwen", "cosy") and not _TTS_READY.is_set()):
            return Response("TTS not ready", status=503, mimetype="text/plain")
        return "True"

    @app.route("/generate_audio", methods=["POST"])
    def generate_audio():
        try:
            item_id = _param("item_id", "1").strip()
            if not item_id:
                return json_response({"success": False, "message": "item_id 不能为空"}, 400)

            topic = _param("topic", "")
            speaker_gender1 = _param("speaker_gender1", "")
            speaker_gender2 = _param("speaker_gender2", "")

            print(f"[流式请求] /generate_audio item_id={item_id} topic={topic!r}")
            state = start_session(item_id, topic, speaker_gender1, speaker_gender2)
            return _stream_headers(
                Response(audio_stream(state), status=200, mimetype="audio/mpeg")
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            return json_response({"success": False, "message": str(exc)}, 500)

    @app.route("/generate_content", methods=["POST"])
    def generate_content():
        try:
            item_id = _param("item_id", "1").strip()
            if not item_id:
                return json_response({"success": False, "message": "item_id 不能为空"}, 400)

            print(f"[流式请求] /generate_content item_id={item_id}")
            state = STORE.wait_for(item_id, config.CONTENT_QUEUE_WAIT_TIMEOUT)
            if state is None:
                return json_response(
                    {"success": False, "message": f"item_id={item_id} 音频流尚未启动，等待超时"},
                    400,
                )
            return _stream_headers(
                Response(content_events(state), status=200, mimetype="text/event-stream")
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            return json_response({"success": False, "message": str(exc)}, 500)

    @app.route("/status/<item_id>", methods=["GET"])
    def session_status(item_id: str):
        """会话生命周期查询：processing / completed / failed（含失败原因）。"""
        state = STORE.get(item_id)
        if state is None:
            return json_response({"success": False, "message": f"会话不存在: {item_id}"}, 404)
        payload = {
            "success": True,
            "item_id": item_id,
            "status": state.status,
            "finished": state.finished,
            "elapsed": round(time.time() - state.created_at, 2),
        }
        if state.error:
            payload["error"] = state.error
        return json_response(payload)

    @app.route("/download/<file_type>/<file_id>", methods=["GET"])
    def download_file(file_type: str, file_id: str):
        print(f"[文件下载] 请求: {file_type}/{file_id}")
        if file_type == "audio":
            path = os.path.join(config.AUDIO_DIR, f"{file_id}.mp3")
            if not os.path.exists(path):
                return json_response(
                    {"success": False, "message": f"音频文件不存在: {file_id}"}, 404
                )
            return send_file(path, mimetype="audio/mpeg", as_attachment=False,
                             download_name=f"{file_id}.mp3")
        if file_type == "image":
            path = os.path.join(config.IMAGE_DIR, f"{file_id}.png")
            if not os.path.exists(path):
                return json_response(
                    {"success": False, "message": f"图片文件不存在: {file_id}"}, 404
                )
            return send_file(path, mimetype="image/png", as_attachment=False,
                             download_name=f"{file_id}.png")
        return json_response({"success": False, "message": f"不支持的文件类型: {file_type}"}, 400)

    return app


app = create_app()

if config.TTS_PRELOAD:
    threading.Thread(target=_preload_tts, daemon=True).start()


def main() -> None:
    print("=" * 60)
    print("AI 播客生成 SUT 服务")
    print("=" * 60)
    print(f"  providers: script={config.SCRIPT_PROVIDER} "
          f"tts={config.TTS_PROVIDER} cover={config.COVER_PROVIDER}")
    print(f"  listening: {config.HOST}:{config.PORT}")
    print("=" * 60)
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
