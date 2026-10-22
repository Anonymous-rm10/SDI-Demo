import ast
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voice_agent.config import ROOT
from voice_agent.jev import arguments
from voice_agent.paths import DEFAULT_PATHS, project_path, use_source


class PortabilityTests(unittest.TestCase):
    def test_bundled_runtime_imports_have_local_dependencies(self):
        packages = {
            "qwen_asr": ROOT / "third_party/Qwen3-ASR",
            "cosyvoice": ROOT / "third_party/CosyVoice",
            "trainjev": ROOT / "third_party/train-jev",
            "matcha": ROOT / "third_party/CosyVoice/third_party/Matcha-TTS",
            "fireredasr2s": ROOT / "third_party/FireRedASR2S",
        }
        for package, base in packages.items():
            for source in (base / package).rglob("*.py"):
                parts = list(source.relative_to(base).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                else:
                    parts = parts[:-1]
                current_package = ".".join(parts)
                for node in ast.walk(ast.parse(source.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        name = "." * node.level + (node.module or "")
                        names = [importlib.util.resolve_name(name, current_package) if node.level else name]
                    elif isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    else:
                        continue
                    for name in names:
                        owner = name.split(".")[0]
                        if owner in packages:
                            target = packages[owner].joinpath(*name.split("."))
                            self.assertTrue(target.with_suffix(".py").is_file()
                                            or target.with_suffix(".pyx").is_file() or target.is_dir(),
                                            f"{source.relative_to(ROOT)} imports missing module {name}")

    def test_default_paths_are_project_relative_from_other_working_directory(self):
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                os.chdir(directory)
                for key, relative in DEFAULT_PATHS.items():
                    self.assertEqual(project_path(key), ROOT / relative)
        finally:
            os.chdir(previous)

    def test_relative_and_absolute_path_overrides(self):
        with patch.dict(os.environ, {"ASR_MODEL": "custom/asr"}):
            self.assertEqual(project_path("ASR_MODEL"), ROOT / "custom/asr")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ASR_MODEL": directory}):
            self.assertEqual(project_path("ASR_MODEL"), Path(directory))

    def test_model_packages_resolve_to_bundled_source(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "path", list(sys.path)):
            for key, module in [("ASR_REPO", "qwen_asr"), ("TTS_REPO", "cosyvoice"), ("JEV_REPO", "trainjev")]:
                repo = use_source(key)
                spec = importlib.util.find_spec(module)
                self.assertIsNotNone(spec)
                self.assertTrue(Path(spec.origin).is_relative_to(repo))
            matcha = importlib.util.find_spec("matcha")
            self.assertTrue(Path(matcha.origin).is_relative_to(ROOT / "third_party/CosyVoice/third_party/Matcha-TTS"))

    def test_jev_entrypoint_uses_bundled_models_and_port_override(self):
        with patch.dict(os.environ, {"JEV_PORT": "8120"}, clear=True):
            args = arguments(["--device", "cpu"])
            self.assertEqual(args[args.index("--base") + 1], str(ROOT / "models/Qwen3.5-4B"))
            self.assertEqual(args[args.index("--adapter") + 1], str(ROOT / "models/train-jev-4b-lora"))
            self.assertIn("8120", args)
            self.assertEqual(args[-2:], ["--device", "cpu"])
