"""
Unit tests for llm_query_system.py

Run with:
    python -m pytest tests/ -v
  or
    python -m unittest discover tests/
"""

import base64
import configparser
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make the project root importable regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).parent.parent))

import llm_query_system as lqs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_ini(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def _make_response(
    status_code: int = 200,
    json_data: dict | None = None,
    content: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Return a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content or (json.dumps(json_data).encode() if json_data else b"")
    resp.text = (json_data and json.dumps(json_data)) or content.decode("utf-8", errors="replace")
    resp.headers = {"Content-Type": content_type}

    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")

    resp.raise_for_status = MagicMock()
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# load_config
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadConfig(unittest.TestCase):

    def _cfg(self, tmp_path, content):
        p = tmp_path / "cfg.ini"
        _write_ini(p, content)
        return lqs.load_config(str(p))

    def test_minimal_valid_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.ini"
            _write_ini(p, "[service]\nname=ollama\ntype=llm\n[endpoint]\nurl=http://localhost:11434/api/generate\n")
            cfg = lqs.load_config(str(p))
        self.assertEqual(cfg["service_name"], "ollama")
        self.assertEqual(cfg["service_type"], "llm")
        self.assertEqual(cfg["endpoint_url"], "http://localhost:11434/api/generate")

    def test_missing_service_section_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.ini"
            _write_ini(p, "[endpoint]\nurl=http://localhost:11434/api/generate\n")
            with self.assertRaises(ValueError):
                lqs.load_config(str(p))

    def test_missing_endpoint_section_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.ini"
            _write_ini(p, "[service]\nname=ollama\ntype=llm\n")
            with self.assertRaises(ValueError):
                lqs.load_config(str(p))

    def test_empty_url_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.ini"
            _write_ini(p, "[service]\nname=ollama\ntype=llm\n[endpoint]\nurl=\n")
            with self.assertRaises(ValueError):
                lqs.load_config(str(p))

    def test_invalid_type_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.ini"
            _write_ini(
                p,
                "[service]\nname=ollama\ntype=banana\n[endpoint]\nurl=http://x\n",
            )
            with self.assertRaises(ValueError):
                lqs.load_config(str(p))

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            lqs.load_config("/no/such/file.ini")

    def test_type_inferred_from_service_name(self):
        """When [service] type is omitted it should be inferred from name."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.ini"
            _write_ini(
                p,
                "[service]\nname=stable_diffusion\n[endpoint]\nurl=http://x\n",
            )
            cfg = lqs.load_config(str(p))
        self.assertEqual(cfg["service_type"], "image")

    def test_parameters_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.ini"
            _write_ini(
                p,
                "[service]\nname=ollama\ntype=llm\n[endpoint]\nurl=http://x\n"
                "[parameters]\ntemperature=0.9\nmax_tokens=256\n",
            )
            cfg = lqs.load_config(str(p))
        self.assertEqual(cfg["parameters"]["temperature"], "0.9")
        self.assertEqual(cfg["parameters"]["max_tokens"], "256")

    def test_model_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.ini"
            _write_ini(
                p,
                "[service]\nname=ollama\ntype=llm\n[endpoint]\nurl=http://x\n"
                "[model]\nname=mistral\n",
            )
            cfg = lqs.load_config(str(p))
        self.assertEqual(cfg["model"], "mistral")


# ─────────────────────────────────────────────────────────────────────────────
# read_input_file
# ─────────────────────────────────────────────────────────────────────────────

class TestReadInputFile(unittest.TestCase):

    def test_reads_txt(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prompt.txt"
            p.write_text("Hello world", encoding="utf-8")
            result = lqs.read_input_file(str(p))
        self.assertEqual(result, "Hello world")

    def test_reads_md(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prompt.md"
            p.write_text("# Title\nContent here", encoding="utf-8")
            result = lqs.read_input_file(str(p))
        self.assertEqual(result, "# Title\nContent here")

    def test_strips_whitespace(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prompt.txt"
            p.write_text("  \n  hello  \n  ", encoding="utf-8")
            result = lqs.read_input_file(str(p))
        self.assertEqual(result, "hello")

    def test_unsupported_extension_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prompt.pdf"
            p.write_text("text", encoding="utf-8")
            with self.assertRaises(ValueError):
                lqs.read_input_file(str(p))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            lqs.read_input_file("/no/such/file.txt")


# ─────────────────────────────────────────────────────────────────────────────
# detect_format
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectFormat(unittest.TestCase):

    def _cfg(self, name, fmt="auto"):
        return {"service_name": name, "request_format": fmt}

    def test_explicit_json(self):
        self.assertEqual(lqs.detect_format(self._cfg("ollama", "json")), "json")

    def test_explicit_text(self):
        self.assertEqual(lqs.detect_format(self._cfg("ollama", "text")), "text")

    def test_auto_ollama(self):
        self.assertEqual(lqs.detect_format(self._cfg("ollama", "auto")), "json")

    def test_auto_llama_cpp(self):
        self.assertEqual(lqs.detect_format(self._cfg("llama_cpp", "auto")), "json")

    def test_auto_stable_diffusion(self):
        self.assertEqual(lqs.detect_format(self._cfg("stable_diffusion", "auto")), "json")

    def test_auto_chatterbox(self):
        self.assertEqual(lqs.detect_format(self._cfg("chatterbox", "auto")), "json")

    def test_auto_unknown_service_defaults_to_json(self):
        self.assertEqual(lqs.detect_format(self._cfg("my_custom_service", "auto")), "json")


# ─────────────────────────────────────────────────────────────────────────────
# build_request
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildRequest(unittest.TestCase):

    def _base_config(self, service_name, service_type="llm", model="", extra_params=None):
        return {
            "service_name":   service_name,
            "service_type":   service_type,
            "endpoint_url":   "http://localhost",
            "model":          model,
            "request_format": "auto",
            "parameters":     extra_params or {},
        }

    def test_text_format_returns_raw_prompt(self):
        cfg = self._base_config("ollama")
        payload, headers, is_json = lqs.build_request("hello", cfg, "text")
        self.assertEqual(payload, "hello")
        self.assertFalse(is_json)
        self.assertIn("text/plain", headers["Content-Type"])

    def test_ollama_json_payload(self):
        cfg = self._base_config("ollama", model="llama3")
        payload, headers, is_json = lqs.build_request("hello", cfg, "json")
        self.assertTrue(is_json)
        self.assertEqual(payload["model"], "llama3")
        self.assertEqual(payload["prompt"], "hello")
        self.assertIn("options", payload)
        self.assertIn("application/json", headers["Content-Type"])

    def test_llama_cpp_json_payload(self):
        cfg = self._base_config("llama_cpp", extra_params={"max_tokens": "256"})
        payload, _, is_json = lqs.build_request("hi", cfg, "json")
        self.assertTrue(is_json)
        self.assertEqual(payload["prompt"], "hi")
        self.assertEqual(payload["n_predict"], 256)

    def test_stable_diffusion_payload(self):
        cfg = self._base_config("stable_diffusion", service_type="image",
                                extra_params={"steps": "30", "width": "768", "height": "768"})
        payload, _, is_json = lqs.build_request("a cat", cfg, "json")
        self.assertTrue(is_json)
        self.assertEqual(payload["prompt"], "a cat")
        self.assertEqual(payload["steps"], 30)
        self.assertEqual(payload["width"], 768)

    def test_stable_diffusion_model_override(self):
        cfg = self._base_config("stable_diffusion", service_type="image", model="v1-5-pruned")
        payload, _, _ = lqs.build_request("a dog", cfg, "json")
        self.assertIn("override_settings", payload)
        self.assertEqual(payload["override_settings"]["sd_model_checkpoint"], "v1-5-pruned")

    def test_chatterbox_payload(self):
        cfg = self._base_config("chatterbox", service_type="tts",
                                extra_params={"exaggeration": "0.8"})
        payload, _, is_json = lqs.build_request("say this", cfg, "json")
        self.assertTrue(is_json)
        self.assertEqual(payload["text"], "say this")
        self.assertAlmostEqual(payload["exaggeration"], 0.8)

    def test_coqui_tts_payload(self):
        cfg = self._base_config("coqui_tts", service_type="tts",
                                extra_params={"speaker_id": "speaker_0", "language_id": "en"})
        payload, _, is_json = lqs.build_request("hello", cfg, "json")
        self.assertTrue(is_json)
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(payload["speaker_id"], "speaker_0")
        self.assertEqual(payload["language_id"], "en")

    def test_unknown_service_generic_fallback(self):
        cfg = self._base_config("my_custom_service", model="custom_model")
        payload, _, is_json = lqs.build_request("test", cfg, "json")
        self.assertTrue(is_json)
        self.assertEqual(payload["prompt"], "test")
        self.assertEqual(payload["model"], "custom_model")

    def test_service_aliases_resolve(self):
        """llama.cpp, llamacpp, and llama_cpp should all produce the same builder."""
        prompt = "hello"
        cfg_base = self._base_config("llama_cpp")
        cfg_dot  = self._base_config("llama.cpp")
        cfg_noul = self._base_config("llamacpp")
        p1, _, _ = lqs.build_request(prompt, cfg_base, "json")
        p2, _, _ = lqs.build_request(prompt, cfg_dot,  "json")
        p3, _, _ = lqs.build_request(prompt, cfg_noul, "json")
        self.assertEqual(p1, p2)
        self.assertEqual(p1, p3)


# ─────────────────────────────────────────────────────────────────────────────
# parse_and_save
# ─────────────────────────────────────────────────────────────────────────────

class TestParseAndSave(unittest.TestCase):

    def _config(self, stype):
        return {"service_type": stype}

    def _logger(self):
        import logging
        logger = logging.getLogger("test_parse_save")
        logger.addHandler(logging.NullHandler())
        return logger

    def test_llm_ollama_response(self):
        resp = _make_response(json_data={"response": "The answer is 42."})
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out = lqs.parse_and_save(resp, self._config("llm"), "testsid", self._logger())
            self.assertTrue(out.exists())
            self.assertIn("42", out.read_text())

    def test_llm_openai_chat_response(self):
        resp = _make_response(json_data={
            "choices": [{"message": {"content": "OpenAI style answer"}}]
        })
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out = lqs.parse_and_save(resp, self._config("llm"), "testsid2", self._logger())
            self.assertIn("OpenAI style answer", out.read_text())

    def test_llm_llama_cpp_response(self):
        resp = _make_response(json_data={"content": "llama content"})
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out = lqs.parse_and_save(resp, self._config("llm"), "testsid3", self._logger())
            self.assertIn("llama content", out.read_text())

    def test_llm_plain_text_fallback(self):
        resp = _make_response(content=b"plain text answer", content_type="text/plain")
        resp.json.side_effect = ValueError("no json")
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out = lqs.parse_and_save(resp, self._config("llm"), "testsid4", self._logger())
            self.assertIn("plain text answer", out.read_text())

    def test_tts_saves_wav(self):
        audio_bytes = b"\x52\x49\x46\x46" + b"\x00" * 40  # minimal RIFF header
        resp = _make_response(content=audio_bytes, content_type="audio/wav")
        resp.json.side_effect = ValueError("no json")
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out = lqs.parse_and_save(resp, self._config("tts"), "testsid5", self._logger())
            self.assertEqual(out.suffix, ".wav")
            self.assertEqual(out.read_bytes(), audio_bytes)

    def test_tts_saves_mp3(self):
        audio_bytes = b"\xff\xfb" + b"\x00" * 10
        resp = _make_response(content=audio_bytes, content_type="audio/mpeg")
        resp.json.side_effect = ValueError("no json")
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out = lqs.parse_and_save(resp, self._config("tts"), "testsid6", self._logger())
            self.assertEqual(out.suffix, ".mp3")

    def test_image_saves_png(self):
        # Create a fake 1×1 pixel PNG in base64
        fake_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20).decode()
        resp = _make_response(json_data={"images": [fake_png]})
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out = lqs.parse_and_save(resp, self._config("image"), "testsid7", self._logger())
            self.assertEqual(out.suffix, ".png")

    def test_image_no_images_raises(self):
        resp = _make_response(json_data={"images": []})
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                with self.assertRaises(ValueError):
                    lqs.parse_and_save(resp, self._config("image"), "testsid8", self._logger())

    def test_image_multiple_saves_all(self):
        fake_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20).decode()
        resp = _make_response(json_data={"images": [fake_png, fake_png, fake_png]})
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out = lqs.parse_and_save(resp, self._config("image"), "multi", self._logger())
            saved = list(Path(d).glob("image_multi*.png"))
        self.assertEqual(len(saved), 3)


# ─────────────────────────────────────────────────────────────────────────────
# write_benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteBenchmark(unittest.TestCase):

    def _logger(self):
        import logging
        logger = logging.getLogger("test_benchmark")
        logger.addHandler(logging.NullHandler())
        return logger

    def test_csv_created_with_header(self):
        resp = _make_response(json_data={"eval_count": 50})
        config = {
            "service_name": "ollama", "service_type": "llm",
            "endpoint_url": "http://localhost", "model": "llama3",
            "_resolved_format": "json",
        }
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                lqs.write_benchmark(
                    "sid123", config, "input.txt",
                    Path(d) / "response_sid123.txt",
                    "test prompt", 1.234, resp, self._logger(),
                )
            csv_path = Path(d) / "benchmark.csv"
            self.assertTrue(csv_path.exists())
            content = csv_path.read_text()
        self.assertIn("session_id", content)
        self.assertIn("sid123", content)
        self.assertIn("ollama", content)

    def test_csv_appended_on_second_call(self):
        resp = _make_response(json_data={"eval_count": 10})
        config = {
            "service_name": "ollama", "service_type": "llm",
            "endpoint_url": "http://localhost", "model": "llama3",
            "_resolved_format": "json",
        }
        with tempfile.TemporaryDirectory() as d:
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                lqs.write_benchmark("sid1", config, "i.txt", Path(d) / "o1.txt",
                                    "p", 1.0, resp, self._logger())
                lqs.write_benchmark("sid2", config, "i.txt", Path(d) / "o2.txt",
                                    "p", 2.0, resp, self._logger())
            lines = (Path(d) / "benchmark.csv").read_text().splitlines()
        # header + 2 data rows
        self.assertEqual(len(lines), 3)


# ─────────────────────────────────────────────────────────────────────────────
# run_query (integration test with mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunQuery(unittest.TestCase):

    def _write_config(self, d, service="ollama", stype="llm",
                      url="http://localhost:11434/api/generate", model="llama3"):
        p = Path(d) / "cfg.ini"
        _write_ini(p,
            f"[service]\nname={service}\ntype={stype}\n"
            f"[endpoint]\nurl={url}\n"
            f"[model]\nname={model}\n"
            "[parameters]\nmax_tokens=64\ntemperature=0.7\n"
        )
        return str(p)

    def _write_input(self, d, text="Tell me a joke."):
        p = Path(d) / "prompt.txt"
        p.write_text(text, encoding="utf-8")
        return str(p)

    @patch("llm_query_system.requests.post")
    def test_llm_query_end_to_end(self, mock_post):
        resp = _make_response(json_data={"response": "Why did the chicken cross the road?"})
        mock_post.return_value = resp

        with tempfile.TemporaryDirectory() as d:
            cfg = self._write_config(d)
            inp = self._write_input(d)
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out_path, sid = lqs.run_query(cfg, inp)
            self.assertTrue(out_path.exists())
            self.assertIn("chicken", out_path.read_text())

    @patch("llm_query_system.requests.post")
    def test_image_query_end_to_end(self, mock_post):
        fake_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20).decode()
        resp = _make_response(json_data={"images": [fake_png]})
        mock_post.return_value = resp

        with tempfile.TemporaryDirectory() as d:
            cfg = self._write_config(d, service="stable_diffusion", stype="image",
                                     url="http://localhost:7860/sdapi/v1/txt2img", model="")
            inp = self._write_input(d, "a beautiful mountain")
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                out_path, sid = lqs.run_query(cfg, inp)

        self.assertEqual(out_path.suffix, ".png")

    @patch("llm_query_system.requests.post")
    def test_benchmark_csv_created(self, mock_post):
        resp = _make_response(json_data={"response": "answer"})
        mock_post.return_value = resp

        with tempfile.TemporaryDirectory() as d:
            cfg = self._write_config(d)
            inp = self._write_input(d)
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                lqs.run_query(cfg, inp)
            self.assertTrue((Path(d) / "benchmark.csv").exists())

    @patch("llm_query_system.requests.post")
    def test_session_log_created(self, mock_post):
        resp = _make_response(json_data={"response": "log test"})
        mock_post.return_value = resp

        with tempfile.TemporaryDirectory() as d:
            cfg = self._write_config(d)
            inp = self._write_input(d)
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                _, sid = lqs.run_query(cfg, inp)
            log_files = list(Path(d).glob("session_*.log"))
        self.assertEqual(len(log_files), 1)

    @patch("llm_query_system.requests.post")
    def test_log_callback_called(self, mock_post):
        resp = _make_response(json_data={"response": "callback test"})
        mock_post.return_value = resp

        messages = []
        with tempfile.TemporaryDirectory() as d:
            cfg = self._write_config(d)
            inp = self._write_input(d)
            with patch.object(lqs, "OUTPUT_DIR", Path(d)):
                lqs.run_query(cfg, inp, log_callback=messages.append)

        self.assertTrue(len(messages) > 0)
        combined = "".join(messages)
        self.assertIn("Session", combined)


# ─────────────────────────────────────────────────────────────────────────────
# Misc / edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_param_helper_coerces_int(self):
        params = {"steps": "25"}
        self.assertEqual(lqs._param(params, "steps", 20), 25)

    def test_param_helper_coerces_float(self):
        params = {"temperature": "0.9"}
        self.assertAlmostEqual(lqs._param(params, "temperature", 0.7), 0.9)

    def test_param_helper_uses_default_for_missing(self):
        self.assertEqual(lqs._param({}, "missing_key", 42), 42)

    def test_param_helper_uses_default_for_bad_value(self):
        params = {"steps": "not_a_number"}
        self.assertEqual(lqs._param(params, "steps", 10), 10)

    def test_session_id_format(self):
        sid = lqs._session_id()
        self.assertRegex(sid, r"^\d{8}_\d{6}_\d")

    def test_extract_llm_text_ollama(self):
        resp = _make_response(json_data={"response": "ollama answer"})
        self.assertEqual(lqs._extract_llm_text(resp), "ollama answer")

    def test_extract_llm_text_openai_chat(self):
        resp = _make_response(json_data={
            "choices": [{"message": {"content": "chat answer"}}]
        })
        self.assertEqual(lqs._extract_llm_text(resp), "chat answer")

    def test_extract_llm_text_openai_text(self):
        resp = _make_response(json_data={
            "choices": [{"text": "completion answer"}]
        })
        self.assertEqual(lqs._extract_llm_text(resp), "completion answer")

    def test_extract_llm_text_llama_cpp_content(self):
        resp = _make_response(json_data={"content": "llama answer"})
        self.assertEqual(lqs._extract_llm_text(resp), "llama answer")

    def test_extract_llm_text_fallback_json(self):
        resp = _make_response(json_data={"weird_key": "value"})
        result = lqs._extract_llm_text(resp)
        self.assertIn("weird_key", result)

    def test_extract_llm_text_no_json(self):
        resp = _make_response(content=b"raw text", content_type="text/plain")
        resp.json.side_effect = ValueError("no json")
        result = lqs._extract_llm_text(resp)
        self.assertEqual(result, "raw text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
