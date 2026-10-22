# FireRed RL shift -> Qwen3-ASR CompassOPSD

该目录保留 `qwen_compassopd_firered` 的 Compass 分布迁移思想，但用 OPSD 的
教师独享参考转录和广义 JSD 训练学生。原 CompassOPD 目录保持为独立对照组。

## 目标

四个策略与 CompassOPD 相同：FireRed RL 强教师 `T`、禁用 RL adapter 的
released FireRed 参考 `R`、当前 Qwen 学生 `pi`、冻结初始 Qwen 参考 `pi_ref`。
学生仅看音频并进行 on-policy 采样；`T` 和 `R` 都看到同一条数据参考转录，学生及
`pi_ref` 都看不到它。在共享普通词表和 `<|im_end|>` 上构造：

```text
teacher_shift = log p(T | audio, reference, prefix)
              - log p(R | audio, reference, prefix)

q = softmax(log p(pi_ref | audio, prefix) + teacher_shift / alpha)
```

令 `s = p(pi | audio, prefix)`，`m = beta*q + (1-beta)*s`，训练损失为：

```text
beta * KL(q || m) + (1-beta) * KL(s || m)
```

`BETA=0` 为 `KL(q || s)`，`BETA=1` 为 `KL(s || q)`，默认 `0.5`。这里遵循现有
OPSD 实现，不额外乘 `alpha` 或温度平方；`alpha` 只控制 teacher shift 注入 target
的强度。因此它是 Compass+OPSD 混合目标，不是原 CompassOPD reverse-KL 目标的
严格等价式。

## 启动

该训练默认对当前 Qwen、冻结 Qwen reference 和 FireRed 教师 LLM 都启用
FlashAttention 2。请先在启动脚本使用的同一个 Python 环境中安装：

```bash
/home/ln10/Workspaces/miniconda3/envs/qwen/bin/python -m pip install \
  flash-attn --no-build-isolation
```

```bash
cd /home/ln10/Workspaces/code/FireRedASR2S
TEACHER_DEVICE=cuda:1 \
bash examples_train/qwen_compassopsd_firered/run_compassopsd.sh
```

单步冒烟训练：

```bash
REPORT_TO=none MAX_STEPS=1 GRAD_ACC=1 NUM_WORKERS=0 \
MAX_COMPLETION_LENGTH=8 OUTPUT_DIR=/tmp/qwen-compassopsd-smoke \
bash examples_train/qwen_compassopsd_firered/run_compassopsd.sh
```

主要环境变量：

| 变量 | 默认值 / 含义 |
| --- | --- |
| `ANCHOR_ALPHA` | `0.5`；Compass target 的 reference anchor 强度 |
| `BETA` | `0.5`；广义 JSD 权重 |
| `TEACHER_SHIFT_CLIP` | `0`；正数时对 teacher shift 对称裁剪 |
| `TEMPERATURE` | `1.0` |
| `TEACHER_REFERENCE_TEMPLATE` | 教师独享提示，必须包含 `{reference}` |
| `STUDENT_REFERENCE_MODEL_DIR` | 冻结学生参考，默认等于学生训练起点 |
| `SFT_COEFFICIENT` | `0.0`；可选参考转录 CE |
| `TOPK` | 空；非空时采用学生 top-k 词表近似 |
| `USE_FLASH_ATTENTION` | `1`；对三个 Transformer 启用 FlashAttention 2，设为 `0` 可关闭 |

断点续训时不要移动 reference anchor：保持 `STUDENT_MODEL_DIR` 和
`STUDENT_REFERENCE_MODEL_DIR` 为首次训练的原始起点，仅用 `RESUME_FROM` 指向
Trainer checkpoint。

日志中的 `compass_jsd` 是未缩放的 JSD，`compassopsd_loss` 是实际蒸馏项；另有
`teacher_shift_abs`、`teacher_reference_kl`、`student_reference_kl` 和
`compass_target_entropy`。模型选择仍应使用独立 ASR 推理得到的 CER/MER。

## CPU 测试

```bash
PYTHONPATH=/home/ln10/Workspaces/code/FireRedASR2S:/home/ln10/Workspaces/code/Qwen3-ASR \
/home/ln10/miniconda3/envs/asr/bin/python \
  examples_train/qwen_compassopsd_firered/test_compassopsd.py -v
```
