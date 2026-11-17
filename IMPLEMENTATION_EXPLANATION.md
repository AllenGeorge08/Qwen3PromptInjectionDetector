# Implementation Explanation

This document provides an in-depth walkthrough of the major implementation decisions in the Qwen3 Prompt Injection Detector, with a focus on the differences between V1 and V2, the training dynamics, and how to interpret the loss curves.

---

## 1. Data Pipeline

### Dataset

The project uses `neuralchemy/Prompt-injection-dataset` with the `full` configuration. Each row contains:

| Column | Description |
|---|---|
| `text` | The input prompt to classify |
| `label` | `0` = benign, `1` = malicious |
| `category` | Attack category (e.g., `direct_injection`, `jailbreak`, `benign`) |
| `source` | Source of the example |
| `severity` | Threat severity (e.g., `low`, `medium`, `high`) |

### Split usage (V1 vs V2)

| Split | V1 | V2 |
|---|---|---|
| Train | ✅ 14,036 rows | ✅ 14,036 rows |
| Validation | ❌ Not used | ✅ 941 rows |
| Test | ✅ Held-out (but also used as eval during training) | ✅ Strictly held-out for final evaluation |

**Why this matters:** V1 passed `test_dataset` as `eval_dataset` to the trainer, meaning the model was evaluated on test data *during* training. This leaks information from the held-out set into the training loop and makes the validation loss optimistic. V2 fixes this by using a proper validation split and reserving the test set for a single final evaluation after training completes.

### Column filtering

Both versions drop all columns except `text`, `label`, `category`, `source`, and `severity` to keep the dataset lean and avoid accidental leakage of metadata into the model input.

---

## 2. Prompt Formatting

### V1: Hand-rolled template

V1 constructs each training example by concatenating raw special-token strings:

```python
prompt_template = lambda text, label, category, severity: f''' Vanderbilt{text} Vanderbilt\n''' + f"{label}" + f"\n{category}" + f"\n{severity}" + "rière"
```

This manually injects ``, ` Vanderbilt`, and `rière` (the Qwen BOS/user/assistant/EOS tokens) as plain text. Risks:
- Tokenizer may re-tokenize or split these special tokens unpredictably.
- No structured system message to guide the model's classification behavior.
- TRL's `SFTTrainer` may add its own special tokens on top, creating duplicate or conflicting tokens.

The system instruction is a single sentence prepended to the text:
> "You're a helpful,respectful and knowledgeable ai assistant. Your job is to classify prompts that contain threats..."

### V2: Qwen-native chat template

V2 replaces the hand-rolled template with `tokenizer.apply_chat_template()`, producing a proper system-user-assistant conversation:

```python
messages = [
    {"role": "system", "content": <detailed classification guidelines>},
    {"role": "user", "content": text},
    {"role": "assistant", "content": f"{label}\n{category}\n{severity}"}
]
formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

Key improvements:
- **Proper special-token handling:** The tokenizer manages BOS, EOS, and chat-role tokens correctly.
- **Rich system prompt:** Explicitly defines 0 = benign vs 1 = malicious, lists attack categories, warns against flagging harmless discussion of harmful topics, and instructs the model to focus on intent to manipulate the AI system.
- **`dataset_kwargs={"add_special_tokens": False}`:** Prevents TRL from injecting extra special tokens that would conflict with the custom chat template.

During inference, V2 uses `add_generation_prompt=True` so the model knows to complete the assistant turn after the user message.

---

## 3. Model Loading & Quantization

Both versions load `Qwen/Qwen3-1.7B` with bitsandbytes 4-bit NF4 quantization and double quantization:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # V2 fixed the V1 typo "compute_dtpye"
)
```

### V2-specific fixes

- **Explicit `dtype=torch.float16`:** Ensures the base model falls back to FP16 on T4 GPUs where BF16 may not be optimal.
- **`torch.cuda.is_bf16_supported()` check:** Confirms hardware capability before choosing mixed precision.
- **`model.activation_checkpointing_enable()`:** Replaces the deprecated `gradient_checkpointing_enable()` (removed in transformers v6).

---

## 4. LoRA Configuration

Parameter-Efficient Fine-Tuning freezes the 4-bit quantized base weights and trains only small low-rank adapter matrices.

| Setting | V1 | V2 |
|---|---|---|
| Rank (`r`) | 8 | 8 |
| Alpha | 32 | 32 |
| Dropout | 0.05 | 0.05 |
| Target modules | `["q_proj"]` | `["q_proj", "v_proj"]` |
| Bias | none | none |
| Task type | CAUSAL_LM | CAUSAL_LM |

### Why add `v_proj` in V2?

In Qwen3's attention, `q_proj` generates queries and `v_proj` generates values. Training only `q_proj` (V1) adapts how the model attends to input tokens but leaves the value projections frozen. Adding `v_proj` lets the adapter also reshape what information is carried forward from each token, giving the classifier more expressive power for detecting subtle injection patterns. This roughly doubles the trainable parameter count (from ~917K to ~1.8M, still only ~0.1% of the 1.7B total).

---

## 5. Training Setup

Both versions use TRL's `SFTTrainer` with `SFTConfig`:

```python
SFTConfig(
    output_dir="./qwen-prompt-injection-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # effective batch size = 16
    learning_rate=2e-4,
    logging_steps=20,
    max_steps=500,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    report_to="none",
    dataset_text_field="text",
    max_length=512,
)
```

### Mixed precision

- **V1:** `bf16=True` — BF16 is native to Qwen but can cause gradient-scaler issues on some T4 configurations.
- **V2:** `fp16=True, bf16=False` — FP16 chosen for T4 compatibility. Before `trainer.train()`, all trainable LoRA parameters are explicitly cast to `float32` to prevent BF16 gradient-scaler errors while keeping forward/backward in FP16.

### Gradient checkpointing

V2 uses `activation_checkpointing_enable()` to trade compute for memory, allowing the 1.7B model to fit on a 16 GB T4 GPU alongside the 4-bit quantized weights and optimizer states.

---

## 6. Training Loss Dynamics

### V1 loss curve (from notebook output)

| Step | Train Loss | Val Loss | Mean Token Acc |
|---:|---:|---:|---:|
| 100 | 2.8572 | 2.5554 | 59.6% |
| 200 | 2.6103 | 2.4999 | 60.8% |
| 300 | 2.6179 | 2.4345 | 61.2% |
| 400 | 2.6701 | 2.3985 | 61.6% |
| 500 | 2.5739 | 2.3910 | 61.7% |

Final training loss: **2.6940** over ~26 minutes.

**Interpretation:** V1's losses are very high (>2.0) and stagnant. A causal LM loss of ~2.5 means the model is still quite uncertain about the next token. This reflects several compounding issues:
- Hand-rolled special tokens likely confused the tokenizer/model alignment.
- Only `q_proj` was adapted, limiting expressive capacity.
- BF16 on T4 may have introduced instability.
- No proper validation split made it hard to detect overfitting/underfitting.

### V2 loss curve (from notebook output)

| Step | Train Loss | Val Loss | Mean Token Acc |
|---:|---:|---:|---:|
| 100 | 0.5729 | 0.4784 | 92.26% |
| 200 | 0.4979 | 0.3917 | 93.07% |
| 300 | 0.4625 | 0.3704 | 93.31% |
| 400 | 0.4033 | 0.3604 | 93.40% |
| 500 | 0.4319 | 0.3561 | 93.45% |

Final training loss: **0.5809** (TrainOutput aggregate) over ~41 minutes.
Final validation loss: **0.3561**.
Mean token accuracy: **93.45%**.

**Interpretation:** V2's losses drop rapidly in the first 100 steps (from ~0.57 to ~0.40) and then stabilize. The gap between train and val loss remains small (~0.05–0.08), indicating good generalization without significant overfitting. Mean token accuracy above 93% means the model correctly predicts the next token (label/category/severity) for the vast majority of positions.

**Why V2 converges so much better:**
1. Qwen-native chat template aligns inputs/outputs with how the base model was pretrained.
2. Rich system prompt gives the model a clear classification task definition.
3. `q_proj` + `v_proj` doubles adapter capacity.
4. Proper validation split enables honest monitoring.
5. FP16 + float32 LoRA cast avoids numerical instability.
6. `add_special_tokens=False` prevents token duplication.

---

## 7. Model Saving & Deployment (V2 only)

V2 adds two artifacts that V1 does not produce:

### 7.1 PEFT adapter

```python
trainer.save_model("./qwen-threat-classifier-adapter")
tokenizer.save_pretrained("./qwen-threat-classifier-adapter")
```

Saves only the LoRA adapter weights (~1.8M parameters) plus the tokenizer and chat template. To use, you must load the base Qwen3-1.7B model and attach the adapter via PEFT.

### 7.2 Merged standalone model

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./qwen-threat-classifier-merged", safe_serialization=True)
```

Merges the LoRA weights into the base model weights and saves a complete FP16 model. This can be loaded with a standard `AutoModelForCausalLM.from_pretrained()` call without the PEFT library, simplifying deployment.

---

## 8. Evaluation (V2)

V2 performs batched inference on the held-out test set:

```python
BATCH_SIZE = 4
MAX_NEW_TOKENS = 40
tokenizer.padding_side = "left"  # decoder-only models need left padding for batched generation
```

For each test sample:
1. Build system-user messages (no assistant turn).
2. Apply chat template with `add_generation_prompt=True`.
3. Tokenize the batch with left padding.
4. Greedy decode (`do_sample=False`) up to 40 new tokens.
5. Strip input tokens, decode generated output.
6. Parse the first non-empty line as the label (`0` or `1`); anything else counts as invalid.

### V2 results (940 test samples)

| Metric | Value |
|---|---|
| Accuracy | 0.96 |
| Precision (Benign / Malicious) | 0.96 / 0.96 |
| Recall (Benign / Malicious) | 0.94 / 0.97 |
| F1 (Benign / Malicious) | 0.95 / 0.96 |
| Macro F1 | 0.96 |
| False Positive Rate | ~5.9% (23 / 390) |
| False Negative Rate | ~3.1% (17 / 552) |
| Invalid predictions | 2 / 940 |
| Throughput | ~4.8 samples/sec |
| Total time | ~196 seconds |

Confusion matrix (actual rows, predicted columns):

```
[[367,  23],   # 367 benign correctly identified, 23 benign falsely flagged
 [ 17, 533]]   # 17 malicious missed (false negatives), 533 malicious correctly caught
```

**Security note:** In a security classifier, false negatives (missed attacks) are typically more costly than false positives. V2's false negative rate of ~3.1% is low but not zero; for production use, consider adding a confidence threshold or a secondary rules layer.

---

## 9. V1 vs V2 Summary

| Area | V1 | V2 |
|---|---|---|
| Data splits | Train + test (test used as eval) | Train + validation + test (proper holdout) |
| Prompt format | Hand-rolled raw special tokens | `tokenizer.apply_chat_template()` |
| System prompt | One sentence | Detailed classification guidelines |
| LoRA targets | `q_proj` only | `q_proj`, `v_proj` |
| Mixed precision | BF16 | FP16 |
| Checkpointing | Deprecated `gradient_checkpointing_enable()` | `activation_checkpointing_enable()` |
| Special tokens in TRL | Default (may duplicate) | `add_special_tokens=False` |
| Trainable params cast | None | Float32 before training |
| Final val loss | 2.3910 | 0.3561 |
| Mean token accuracy | 61.7% | 93.45% |
| Model artifacts | None | Adapter + merged standalone model |
| Held-out eval | Single ad-hoc prompt | Full test set with metrics + confusion matrix |

**Bottom line:** V1 is an exploratory first pass that demonstrates the approach can work at all. V2 is the recommended version — it fixes data leakage, uses Qwen's native formatting, expands adapter capacity, stabilizes numerics, saves deployable artifacts, and produces a measurable 0.96 accuracy on a held-out test set.
