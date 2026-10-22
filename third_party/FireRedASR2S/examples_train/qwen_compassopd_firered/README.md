# FireRed RL shift -> Qwen3-ASR CompassOPD

这个目录实现独立的跨家族 CompassOPD 训练，不修改现有
`qwen_opd_firered` 和 `qwen_opsd_firered`。

## 方法

四个策略分别是：

| 角色 | 本实现 |
| --- | --- |
| 强教师 `T` | 加载 RL checkpoint 的 FireRedASR2-LLM |
| 教师参考 `R` | 同一 FireRed 实例禁用 RL LoRA 后的 released policy |
| 当前学生 `pi` | 全参数训练的 Qwen3-ASR |
| 学生参考 `pi_ref` | 训练开始时冻结的 Qwen3-ASR checkpoint |

Qwen 当前策略先生成 on-policy 转录。对每一个学生访问的转录前缀，在经过验证的
共享普通词表以及共享 `<|im_end|>` 上计算：

```text
teacher_shift = log_softmax(FireRed_RL)
              - log_softmax(FireRed_released)

target_logprob = log_softmax(
    log_softmax(Qwen_frozen_reference) + teacher_shift / alpha
)

loss = alpha * KL(Qwen_current || target)
```

乘以 `alpha` 不改变最优 target，并使梯度尺度与论文的
`teacher_shift - alpha * student_displacement` 更新一致。这等价于论文中目标策略：

```text
pi* proportional_to pi_ref * exp(teacher_shift / alpha)
```

所有参考 logits 都会 detach，只有当前 Qwen 更新。概率计算使用 float32，模型
forward 仍使用 BF16/FP16。损失只覆盖学生采样的有效转录和终止位置，不覆盖
FireRed/Qwen 各自不同的 prompt。

## 为什么可以用一个 FireRed 实例

当前 RL 训练从 released FireRed 直接开始，`init_checkpoint` 为空；released rank-64
LoRA 在创建新的 rank-8 RL LoRA 前已经 merge 到基座。因此：

- 启用 RL adapter 得到强教师；
- 禁用 RL adapter 得到 RL 前的 released FireRed；
- speech encoder 输出只计算一次；
- 不需要同时在显存中加载两个 7B 模型。

训练入口会检查 RL checkpoint 的配置。如果 RL 使用了非空 `init_checkpoint`、LLM
全参数训练，或者 RL LoRA 与 released LoRA 结构完全相同，程序会拒绝启动，因为
此时禁用 adapter 不能精确恢复 RL 初始策略。

学生采用全参数训练，因此仍需加载一份约 0.6B 的冻结 Qwen reference。

## 训练

先选择一个在目标域验证集上确实优于 released FireRed 的 RL checkpoint。不要仅按
step 最大来选择。当前脚本默认使用工作区内存在的 `checkpoint-step-00016000`，运行时
可以覆盖：

```bash
cd /home/ln10/Workspaces/code/FireRedASR2S

TEACHER_DEVICE=cuda:1 \
TEACHER_CHECKPOINT=/home/ln10/Workspaces/code/FireRedASR2S/exp/FireRedASR2-LLM-RL/checkpoint-step-00016000 \
TRAIN_FILE=/home/ln10/Workspaces/datasets/WildElder/train_opd.jsonl \
EVAL_FILE=/path/to/heldout.jsonl \
bash examples_train/qwen_compassopd_firered/run_compassopd.sh
```

脚本默认：

```text
alpha=0.5
temperature=1.0
learning_rate=2e-6
batch_size=1
gradient_accumulation=8
SFT coefficient=0
full shared vocabulary
```

单步冒烟测试：

```bash
REPORT_TO=none MAX_STEPS=1 GRAD_ACC=1 NUM_WORKERS=0 \
MAX_COMPLETION_LENGTH=8 \
OUTPUT_DIR=/tmp/qwen-compassopd-smoke \
bash examples_train/qwen_compassopd_firered/run_compassopd.sh
```

启动脚本支持以下环境变量：

| 环境变量 | 默认值 / 作用 |
| --- | --- |
| `STUDENT_MODEL_DIR` | 当前 Qwen 的训练起点 |
| `STUDENT_REFERENCE_MODEL_DIR` | 冻结学生参考；默认与训练起点相同 |
| `TEACHER_CHECKPOINT` | 强教师 RL checkpoint，必须提供有效目录 |
| `ANCHOR_ALPHA` | `0.5`；越小，teacher shift 的作用越强 |
| `TEACHER_SHIFT_CLIP` | `0`，不裁剪；正数表示对 shift 做对称裁剪 |
| `TEMPERATURE` | `1.0` |
| `TOPK` | 空，使用完整共享词表；非空时使用学生 top-k 近似 |
| `SFT_COEFFICIENT` | `0.0`；可选参考转录 CE，不属于原始 Compass 目标 |
| `LR` | `2e-6` |
| `BATCH_SIZE` / `GRAD_ACC` | `1` / `8` |
| `EPOCHS` / `MAX_STEPS` | `1` / `-1` |
| `EVAL_FILE` | 可选；这里评估 sampled Compass loss，不是 CER/MER |
| `REPORT_TO` | `wandb`；可设置为 `none` |

可以把额外 CLI 参数附加在脚本末尾，例如：

```bash
bash examples_train/qwen_compassopd_firered/run_compassopd.sh \
  --language Chinese --seed 123
```

断点续训时，`STUDENT_MODEL_DIR` 和 `STUDENT_REFERENCE_MODEL_DIR` 仍应指向首次训练的
原始学生起点，再通过 `RESUME_FROM` 指向 Trainer checkpoint。不要把 reference 改成
已经训练过的输出目录，否则学生锚点会发生变化。

## 日志指标

- `compass_kl`：当前学生到 Compass target 的 reverse KL；
- `compass_loss`：实际参与训练的 `alpha * compass_kl`；
- `teacher_shift_abs` / `teacher_shift_rms`：RL 相对 released FireRed 的变化大小；
- `teacher_reference_kl`：`KL(FireRed_RL || FireRed_released)`；
- `student_reference_kl`：当前 Qwen 到冻结初始 Qwen 的 KL；
- `completion_length`、`empty_rollout_rate`、`unsupported_special_rate`；
- `sft_loss`：仅在 `SFT_COEFFICIENT > 0` 时参与总损失。

如果 `teacher_shift_abs` 和 `teacher_reference_kl` 长期接近零，说明所选 RL checkpoint
没有提供足够的策略变化；如果它们很大且 loss/梯度不稳定，先增大 `ANCHOR_ALPHA`，
再考虑设置 `TEACHER_SHIFT_CLIP`。

## 推荐对照实验

所有实验使用相同 Qwen 起点、训练样本、训练步数和随机种子：

1. 不训练的 Qwen baseline；
2. released FireRed 标准 OPD；
3. RL FireRed 标准 OPD；
4. 本目录 CompassOPD，`alpha=0.5`；
5. CompassOPD，`alpha=0.25/1.0`。

最终模型必须用独立 ASR 推理脚本计算 CER/MER。Trainer 的 `EVAL_FILE` 只反映蒸馏
目标，不应当作为唯一的 checkpoint 选择依据。

## 测试

CPU 数值和 checkpoint 约束测试：

```bash
cd /home/ln10/Workspaces/code/FireRedASR2S
PYTHONPATH=/home/ln10/Workspaces/code/FireRedASR2S:/home/ln10/Workspaces/code/Qwen3-ASR \
/home/ln10/miniconda3/envs/asr/bin/python \
  examples_train/qwen_compassopd_firered/test_compassopd.py -v
```

测试覆盖目标分布公式、梯度隔离、identity 情形、top-k、数值稳定性以及 RL
checkpoint/reference 的安全检查。
