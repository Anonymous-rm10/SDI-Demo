# FireRed 教师 → Qwen3-ASR OPSD 训练

基于 `../qwen_opd_firered` 的训练流程，保留冻结的 FireRedASR2-LLM 教师和
全参数训练的 Qwen3-ASR 学生，加入教师独享的参考转录及广义 JSD 损失。
这里是跨模型的 OPSD 风格蒸馏，**不是同源模型的严格自蒸馏**。

## 每步训练

1. 学生仅看到原有 `prompt`、音频和固定的 `language ... <asr_text>` 前缀，
   自行采样转录及结束符。数据中的 `prompt` 应为普通任务提示，不应包含答案。
2. 教师在 FireRed 的 user 消息中看到音频、转写指令和该条样本的参考转录。
   参考转录仅用于提示上下文，教师 assistant 部分接入学生采样的原始 token ID。
3. 沿用 OPD 的共享词表校验及逐 token 对齐，只计算有效转录和共享
   `<|im_end|>` 的蒸馏损失，排除提示词、padding 和其他特殊 token。
4. 仅更新学生；教师冻结。无效采样重试后仍无效时，该 micro-batch 返回零损失，
   不用参考转录替换采样结果。可选 `SFT_COEFFICIENT` 加入参考转录 CE。

令学生、教师分布分别为 `s`、`t`，`m = beta*t + (1-beta)*s`：

- `0 < BETA < 1`：`beta*KL(t || m) + (1-beta)*KL(s || m)`。
- `BETA=0`：前向 KL，即 `KL(t || s)`。
- `BETA=1`：反向 KL，即 `KL(s || t)`，但教师仍看到参考转录。

该约定与 [TRL GKD](https://huggingface.co/docs/trl/v0.17.0/gkd_trainer) 及本工作区
ms-swift 的 `gkd_loss.py` 一致。默认 `BETA=0.5`；损失用 float32 和 log-space
混合计算。沿用 OPD 的逐样本有效 token 平均，不额外乘温度平方。
`TOPK` 非空时，双方在学生的 top-k 词表子集上重新归一化，是完整词表损失的近似。

## 启动

```bash
cd /home/ln10/Workspaces/code/FireRedASR2S
TEACHER_DEVICE=cuda:1 \
TRAIN_FILE=/home/ln10/Workspaces/datasets/WildElder/train_opd.jsonl \
bash examples_train/qwen_opsd_firered/run_opsd.sh
```

默认模型、FireRed RL checkpoint 和训练数据沿用 OPD 启动脚本；输出目录为
`/home/ln10/Workspaces/models/Qwen3-ASR-0.6B-opsd-firered_rl`。
需要已安装依赖的 ASR Python 环境和本地 `code/Qwen3-ASR`；启动脚本设置
`PYTHONPATH`。教师为 7B，推荐两张 GPU；仅有一张时教师和学生共享 `cuda:0`。
目前沿用单训练进程，不支持通过 torchrun/DeepSpeed 分布式启动。

单步冒烟训练（仍需加载完整模型）：

```bash
REPORT_TO=none MAX_STEPS=1 GRAD_ACC=1 NUM_WORKERS=0 \
MAX_COMPLETION_LENGTH=8 OUTPUT_DIR=/tmp/qwen-opsd-smoke \
bash examples_train/qwen_opsd_firered/run_opsd.sh
```

输入 JSONL 接受 `audio` / `wav` / `wav_path` 和 `text` / `reference`，
兼容带 `language ... <asr_text>` 的转录文本。相对音频路径按 JSONL 所在目录解析。

常用环境变量：

| 参数 | 默认值 / 作用 |
| --- | --- |
| `BETA` | `0.5`，广义 JSD 权重 |
| `TEMPERATURE` | `1.0`，蒸馏温度 |
| `SAMPLING_TEMPERATURE` / `SAMPLING_TOP_P` | `1.0` / `0.95`，学生采样参数 |
| `SFT_COEFFICIENT` | `0.0`，可选参考转录 CE 权重 |
| `TEACHER_MAX_TEXT_TOKENS` | `1024`，教师文本提示上限（不含展开后的音频及采样）；超长报错，不静默截断参考 |
| `TEACHER_REFERENCE_TEMPLATE` | 默认中文辅助转录提示；自定义时必须包含 `{reference}`，其他字面花括号须写成 `{{` / `}}` |
| `TEACHER_CHECKPOINT` | 沿用 OPD 的 RL checkpoint；显式设为空使用基础 FireRed 权重 |
| `ASR_LANGUAGE` | `Chinese`，学生固定输出协议的语言 |
| `BATCH_SIZE` / `GRAD_ACC` / `LR` | `1` / `1` / `1e-5` |
| `MAX_ROLLOUT_RETRIES` | `3`，首次采样失败后的额外重试次数 |
| `LOG_ROLLOUT_TEXT` | `1`，记录每次原始采样；可设 `0` 关闭 |
| `EVAL_FILE` | 可选验证 JSONL，评估采样蒸馏损失，不是 CER |
| `RESUME_FROM` | 可选 Trainer checkpoint 路径 |
| `REPORT_TO` | `wandb`；可设 `none` |

另支持 OPD 已有的 `STUDENT_MODEL_DIR`、`TEACHER_MODEL_DIR`、`QWEN_PROJECT_DIR`、
`PYTHON_BIN`、`TOPK`、`EPOCHS`、`SAVE_STEPS` 等变量。额外 CLI 参数可直接追加在
脚本末尾，例如 `--teacher_instruction '请转写音频为文字' --seed 123`。
checkpoint 保存完整学生和 processor，可用于原有 Qwen 推理流程。

## CPU 测试

```bash
PYTHONPATH=../Qwen3-ASR \
/home/ln10/miniconda3/envs/asr/bin/python -m unittest discover -s tests -p 'test_qwen_opsd.py' -v
```

测试覆盖 JSD/KL 数值、混合精度极端输入、top-k、教师梯度隔离、变长提示及音频
合并后的响应标签对齐、EOS、学生参数更新、批量 padding 和参数校验。
