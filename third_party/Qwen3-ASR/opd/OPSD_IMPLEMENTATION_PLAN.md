# OPSD 训练实现计划（Qwen3-ASR）

## 目标

在 Qwen3-ASR 项目里实现论文 *On-Policy Self-Distillation for Multi-Dialect ASR* 的 OPSD 训练。

与现有 OPD（`opd/opd_trainer.py`）的**本质区别**：
1. **Teacher 是冻结的 student 副本（自蒸馏）**，不是独立的大模型。
2. **Teacher 在打分时能看到 reference（参考转录）作为 privileged context**，student 看不到。
3. **Loss 用 JSD(β)** 而不是反向 KL。

## 关键约束（已从代码验证）

- **chat template 会把 system 消息文本拼进 system 段，但 user 消息里的文本会被丢弃**，只保留 `type:audio` 的占位符。
  → 参考转录必须放进 **system 消息**，才能成为 teacher 的 privileged 输入。
- 模型接口：`model.forward` 已 patch 到 `thinker.forward`，接受 `input_ids / attention_mask / input_features / feature_attention_mask / labels`。`generate()` 返回 `sequences`。
- 数据格式：`{uttid, reference, wav_path}`，`reference` 即参考转录。

## 实现内容

### 新增文件：`opd/opsd_trainer.py`
`OPSDTrainer`，继承自 Trainer，复用现有基础设施，核心是 `compute_loss`：

1. **Student on-policy rollout**：用 `dataset 的 base prompt + 音频` 让 student `generate()`，得到 on-policy completion。
2. **Teacher 打分**：把同一段 on-policy completion，接在 `teacher_prefix`（base prompt + reference 拼接后的前缀）后面，用冻结的 teacher forward 得到 logits。
3. **JSD loss**：`β` 在 0..1 间控制，`β=0` 前向 KL、`β=1` 反向 KL、`0<β<1` 对称 JSD（对齐 ms-swift 的 `jsd_loss`）。
4. 更新 student 参数；teacher 冻结（`torch.inference_mode()` + `requires_grad=False`）。

在 trainer 里把 teacher 从 student 的 `Qwen3ASRModel.from_pretrained(同 checkpoint)` 加载为**独立实例**并 `eval()`，确保是**同权重、冻结**的 teacher。

### 数据侧：`opd/opsd_data_utils.py`（或并入现有 dataloader 逻辑）
- 为每条数据构造**两个前缀**：
  - `prefix_text`（student 用）：`base_prompt + 音频` 编码后的前缀。
  - `teacher_prefix_text`（teacher 用）：`base_prompt + reference 拼接进 system 消息 + 音频` 编码后的前缀。
- 保持二者**共享同一段音频**。

### `opd/train_opsd.py`
CLI 入口，仿照 `train_opd.py`，参数包含：
- `--model_path`（teacher 与 student 同源，作 self-distillation）
- `--train_file` / `--eval_file`
- `--beta`（JSD 插值，默认 0.5）
- `--temperature`（蒸馏软化，默认 1.0 或 0.8）
- 其余训练参数沿用现有 OPD 逻辑。

### `opd/run_opsd.sh`
仿照 `run_opd.sh`，默认指向 `Qwen3-ASR-0.6B`（作为自蒸馏的师生同源 checkpoint）。**保留一个开关**说明与 1.7B teacher 模式的差异。

## 设计决策（已与用户确认）

1. **保留 0.6B student + 1.7B teacher 的组合**。teacher 是独立的 1.7B 模型，不是 student 自身副本。因此本实现只保留 OPSD 的关键特征：**teacher 打分时看到 privileged reference，loss 用 JSD(β)**，而非严格意义的"self-distillation"。
2. **全参微调**：沿用现有 OPD 的全参 Trainer，不用 LoRA，后续可扩展。
3. **teacher 冻结**：独立副本 + `eval()` + `inference_mode()`，不更新参数。

## 验证方式
- 写完后，用 `--max_completion_length 8`、`--steps 1` 的小规模 smoke test 跑一遍确认 loss 能下降、 teacher logits 能生成、内存无爆。
- 用现有 `eval/` 目录做方言/Mandarin 的 CER 对比（可选）。

## 需要用户确认的取舍
- 如果用户**已有 SFT checkpoint**（论文流程的"SFT 打底"），应把 `--model_path` 指向那个 checkpoint，而不是 base model。这是 OPSD 能够"保留 Mandarin"的前提。
- 是否要支持 `--teacher_model_path` 作为独立教师（偏离论文的 self-distillation）。默认不支持。
