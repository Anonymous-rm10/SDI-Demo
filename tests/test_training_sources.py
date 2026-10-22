import os
import sys
import unittest
from pathlib import Path

from scripts.model import COMPONENTS, ROOT, prepare


class TrainingSourceTests(unittest.TestCase):
    def test_training_entrypoints_and_recipes_are_present(self):
        expected = {
            "qwen3-asr": [
                "finetuning/qwen3_asr_sft.py", "finetuning/run.sh",
                "opd/train_opd.py", "opd/opd_trainer.py", "opd/run_opd.sh",
                "opd/train_opsd.py", "opd/opsd_trainer.py", "opd/opsd_data_utils.py", "opd/run_opsd.sh",
            ],
            "cosyvoice": [
                "cosyvoice/bin/train.py", "cosyvoice/dataset/dataset.py", "cosyvoice/utils/executor.py",
                "examples/libritts/cosyvoice/run.sh", "examples/libritts/cosyvoice2/run.sh",
                "examples/libritts/cosyvoice2/run_dpo.sh", "examples/libritts/cosyvoice3/run.sh",
                "examples/grpo/cosyvoice2/run.sh", "examples/grpo/cosyvoice2/reward_tts.py",
            ],
            "train-jev": [
                "trainjev/train.py", "trainjev/train_full.py", "trainjev/resume.py", "trainjev/losses.py",
                "scripts/run_train.sh", "scripts/run_train_27b.sh", "scripts/prepare_data.sh",
            ],
            "fireredasr2s": [
                "fireredasr2s/fireredasr2/train_sft.py", "fireredasr2s/fireredasr2/train_rl.py",
                "fireredasr2s/fireredasr2/train_qwen_opd.py", "fireredasr2s/fireredasr2/train_qwen_opsd.py",
                "fireredasr2s/fireredasr2/sft/data.py", "fireredasr2s/fireredasr2/rl/grpo.py",
                "fireredasr2s/fireredasr2/rl/semantic_reward.py",
                "examples_train/llm_sft/sft_config.json", "examples_train/llm_rl/rl_config.json",
                "examples_train/qwen_compassopd_firered/train_compassopd.py",
                "examples_train/qwen_compassopsd_firered/train_compassopsd.py",
            ],
            "matcha-tts": [
                "matcha/train.py", "matcha/models/matcha_tts.py", "configs/train.yaml",
                "matcha/utils/monotonic_align/core.pyx", "setup.py",
            ],
        }
        for component, files in expected.items():
            for relative in files:
                with self.subTest(component=component, file=relative):
                    self.assertTrue((COMPONENTS[component] / relative).is_file())

    def test_runner_selects_interpreter_and_local_model_sources(self):
        cwd, command, env = prepare("fireredasr2s", ["python", "-m", "fireredasr2s.fireredasr2.train_sft"])
        self.assertEqual(cwd, COMPONENTS["fireredasr2s"])
        self.assertEqual(command[0], sys.executable)
        for source in COMPONENTS.values():
            self.assertIn(str(source), env["PYTHONPATH"].split(os.pathsep))
        cwd, command, env = prepare("train-jev", ["torchrun", "--nproc_per_node", "2", "-m", "trainjev.train"])
        self.assertEqual(command[:3], [sys.executable, "-m", "torch.distributed.run"])

    def test_recipe_working_directory_and_invalid_requests(self):
        cwd, _, _ = prepare("cosyvoice", ["bash", "run.sh"], "examples/libritts/cosyvoice3")
        self.assertEqual(cwd, ROOT / "third_party/CosyVoice/examples/libritts/cosyvoice3")
        with self.assertRaises(ValueError):
            prepare("qwen3-asr", [])
        with self.assertRaises(ValueError):
            prepare("qwen3-asr", ["python"], "../..")
