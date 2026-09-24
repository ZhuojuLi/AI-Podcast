# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from app.providers import cosy
from app.schemas import Transcript, Turn


class CosyProviderTest(unittest.TestCase):
    def setUp(self):
        root = os.path.dirname(os.path.dirname(__file__))
        refs = [
            os.path.join(root, "resources/voices/male.wav"),
            os.path.join(root, "resources/voices/female.wav"),
        ]
        if not all(os.path.isfile(ref) for ref in refs):
            raise unittest.SkipTest("参考音频未随仓库分发，自备后生效（见 resources/voices/README.md）")
        self.env = patch.dict(os.environ, {
            "TTS_MODEL_PATH": "/models/test-cosy",
            "TTS_MALE_REF": os.path.join(root, "resources/voices/male.wav"),
            "TTS_FEMALE_REF": os.path.join(root, "resources/voices/female.wav"),
            "TTS_FEMALE_TEMPO": "1.07",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_male_and_female_mapping_and_incremental_output(self):
        provider = cosy.CosyTTSProvider()
        fake_model = type("Model", (), {"sample_rate": 24000})()
        calls = []

        def synth(model, turn, index, ref, female):
            calls.append((index, os.path.basename(ref), female))
            return np.zeros(2400, dtype=np.float32), 24000

        def segments():
            yield [Turn(1, "开场。")]
            yield [Turn(2, "这里接着说。")]

        with patch.object(cosy, "_load_model", return_value=fake_model), \
             patch.object(provider, "_synth", side_effect=synth), \
             patch.object(cosy, "_encode_mp3", return_value=b"mp3"):
            stream = provider.stream_turns(segments(), "男", "女")
            self.assertEqual(next(stream), b"mp3")
            self.assertEqual(calls, [(0, "male.wav", False)])
            list(stream)
            self.assertEqual(calls[1], (1, "female.wav", True))

    def test_same_gender_uses_distinct_references(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cases = [
            ({"TTS_MALE_REF2": os.path.join(root, "resources/voices/male2.wav")},
             ("男", "男"), ["male.wav", "male2.wav"], False),
            ({"TTS_FEMALE_REF2": os.path.join(root, "resources/voices/female2.wav")},
             ("女", "女"), ["female.wav", "female2.wav"], True),
        ]
        for extra_env, genders, expected_refs, expected_female in cases:
            with self.subTest(genders=genders):
                with patch.dict(os.environ, extra_env):
                    provider = cosy.CosyTTSProvider()
                fake_model = type("Model", (), {"sample_rate": 24000})()
                calls = []

                def synth(model, turn, index, ref, female):
                    calls.append((index, os.path.basename(ref), female))
                    return np.zeros(2400, dtype=np.float32), 24000

                def segments():
                    yield [Turn(1, "开场。"), Turn(2, "接话。")]

                with patch.object(cosy, "_load_model", return_value=fake_model), \
                     patch.object(provider, "_synth", side_effect=synth), \
                     patch.object(cosy, "_encode_mp3", return_value=b"mp3"):
                    list(provider.stream_turns(segments(), *genders))
                self.assertEqual([c[1] for c in calls], expected_refs)
                self.assertTrue(all(c[2] == expected_female for c in calls))

    def test_missing_second_voice_falls_back_to_primary(self):
        with patch.dict(os.environ, {"TTS_MALE_REF2": "/nonexistent/male2.wav"}):
            provider = cosy.CosyTTSProvider()
        self.assertEqual(provider.male_ref2, provider.male_ref)

    def test_secondary_voices_get_distinct_persona_instructions(self):
        root = os.path.dirname(os.path.dirname(__file__))
        with patch.dict(os.environ, {
            "TTS_MALE_REF2": os.path.join(root, "resources/voices/male2.wav"),
            "TTS_FEMALE_REF2": os.path.join(root, "resources/voices/female2.wav"),
        }):
            provider = cosy.CosyTTSProvider()

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

        class FakeModel:
            sample_rate = 24000

            def __init__(self):
                self.prompts = []

            def inference_instruct2(self, text, prompt, ref, **kwargs):
                self.prompts.append(prompt)
                return [{"tts_speech": FakeTensor()}]

        model = FakeModel()
        fake_audio = type("Audio", (), {
            "squeeze": lambda self, dim: self,
            "float": lambda self: self,
            "numpy": lambda self: np.zeros(24000 * 4, dtype=np.float32),
        })()
        fake_torch = types.SimpleNamespace(cat=lambda chunks, dim=-1: fake_audio)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            provider._synth(model, Turn(1, "我们继续看这个问题。"), 1,
                            provider.male_ref, False)
            provider._synth(model, Turn(2, "我换一个角度补充。"), 2,
                            provider.male_ref2, False)
            provider._synth(model, Turn(2, "这背后还有一个细节。"), 3,
                            provider.female_ref2, True)

        # 时长校验可能触发重采样，同一次 _synth 会留下多条 prompt，
        # 因此用 any() 判断人设指令是否注入，而不是按下标。
        self.assertFalse(any("健谈的知识型搭档" in p for p in model.prompts[:1]))
        self.assertTrue(any("健谈的知识型搭档" in p for p in model.prompts))
        self.assertTrue(any("访谈主持" in p for p in model.prompts))

    def test_female_tempo_shortens_audio_without_pitch_shift(self):
        sr = 24000
        original = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
        faster = cosy._change_tempo(original, sr, 1.07)
        self.assertAlmostEqual(len(faster) / len(original), 1 / 1.07, delta=0.02)
        spectrum = np.abs(np.fft.rfft(faster))
        peak_hz = np.fft.rfftfreq(len(faster), 1 / sr)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak_hz, 220, delta=3)
        self.assertTrue(np.array_equal(cosy._change_tempo(original, sr, 1.0), original))

    def test_transcript_interface(self):
        provider = cosy.CosyTTSProvider()
        transcript = Transcript("测试", [Turn(1, "你好。"), Turn(2, "你好。")])
        with patch.object(provider, "stream_turns", return_value=iter([b"abc"])) as stream:
            self.assertEqual(list(provider.stream_mp3(transcript, "男", "女")), [b"abc"])
            self.assertEqual(stream.call_args.args[1:], ("男", "女"))

    def test_encoded_audio_uses_contract_sample_rate(self):
        from app.providers.tts import _encode_mp3

        wav = np.zeros(24000, dtype=np.float32)
        encoded = _encode_mp3(wav, 24000, with_id3=True)
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-f", "mp3", "-i", "pipe:0",
             "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0"],
            input=encoded, capture_output=True, check=True,
        )
        self.assertEqual(result.stdout.decode().strip(), "16000,1")

    def test_early_id3_chunk_is_immediate_and_audio_remains_decodable(self):
        from app.providers.tts import _encode_mp3

        with patch.dict(os.environ, {"TTS_EARLY_ID3_KIB": "1024"}):
            provider = cosy.CosyTTSProvider()
        fake_model = type("Model", (), {"sample_rate": 24000})()
        wav = np.zeros(24000, dtype=np.float32)
        stream = provider.stream_turns(iter([[Turn(1, "你好。")]]), "男", "女")

        # 首块不等待文稿或 TTS；内容是有效 ID3 元数据，而非无效空字节。
        header = next(stream)
        self.assertEqual(header[:3], b"ID3")
        self.assertEqual(len(header), 1024 * 1024 + 10)
        with patch.object(cosy, "_load_model", return_value=fake_model), \
             patch.object(provider, "_synth", return_value=(wav, 24000)):
            audio = header + b"".join(stream)
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "mp3", "-i", "pipe:0",
             "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"],
            input=audio, capture_output=True, check=True,
        )
        duration = len(decoded.stdout) / (16000 * 4)
        self.assertGreater(duration, 0.9)
        self.assertLess(duration, 1.5)
        with tempfile.NamedTemporaryFile(suffix=".mp3") as output:
            output.write(audio)
            output.flush()
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-i", output.name,
                 "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"],
                capture_output=True, check=True,
            )
            self.assertGreater(float(probe.stdout), 0.9)
            self.assertLess(float(probe.stdout), 1.5)


if __name__ == "__main__":
    unittest.main()
