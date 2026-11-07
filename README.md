# Qwen3 Prompt Injection Detector

A fine-tuned LLM-based security classifier built using **Qwen3-1.7B and QLoRA** to detect prompt injection and jailbreak attempts.

As LLMs become integrated into applications, they increasingly interact with untrusted user inputs, external documents, and retrieved content. These inputs can contain malicious instructions designed to override system prompts, manipulate model behavior, or bypass application-level safeguards.

This project explores parameter-efficient fine-tuning of a small language model to identify potentially malicious inputs and classify them by **label, attack category, and severity**.

Rather than relying on a general-purpose LLM to detect threats through prompting alone, the project fine-tunes Qwen3-1.7B on a labeled prompt-injection dataset using 4-bit quantization and Low-Rank Adaptation (LoRA).

## Results at a glance

### Training

| Metric | Value |
|---|---:|
| Base model | Qwen3-1.7B |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA rank | 8 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Training steps | 500 |
| Training time | ~25 minutes |
| Trainable parameters | 917,504 |
| Trainable parameter percentage | ~0.0533% |
| Final validation loss | 1.2916 |
| Final mean token accuracy | 79.28% |

### Training progress

Validation loss decreased consistently throughout the training run, from `1.4768` at step 100 to `1.2916` at step 500.

![Training Results](assets/training-results.png)

### Threat detection evaluation

The model was evaluated on a separate held-out test dataset.

| Metric | Result |
|---|---:|
| Accuracy | `ADD VALUE` |
| Precision | `ADD VALUE` |
| Recall | `ADD VALUE` |
| Macro F1 | `ADD VALUE` |
| False Positive Rate | `ADD VALUE` |
| False Negative Rate | `ADD VALUE` |

![Confusion Matrix](assets/confusion-matrix.png)

*Note: Mean token accuracy is a language-model training metric, not the final threat-classification accuracy. The classification metrics above measure actual detection performance.*

## Features

- Fine-tuning of **Qwen3-1.7B** for prompt injection and jailbreak detection.
- Parameter-efficient training using **QLoRA**.
- 4-bit NF4 quantization with double quantization.
- LoRA adapters to update a small fraction of model parameters.
- Structured classification output containing:
  - **Label:** benign or malicious.
  - **Category:** attack category.
  - **Severity:** predicted threat severity.
- Dataset filtering and preprocessing using Hugging Face Datasets.
- Qwen-native conversational formatting through `tokenizer.apply_chat_template()`.
- Validation-based training monitoring.
- Held-out test evaluation.
- Classification reports and confusion matrix.
- Prediction-level error analysis.

## Model and dataset

### Base model

[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

Qwen3-1.7B serves as the base language model. Its weights are loaded in 4-bit precision, while trainable LoRA adapters learn the threat-classification task.

### Dataset

[neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset)

The project uses the `full` configuration of the Prompt Injection Dataset.

The dataset contains labeled examples covering benign inputs and multiple prompt injection attack categories.

Dataset splits:

| Split | Purpose |
|---|---|
| Training | Fine-tuning the model |
| Validation | Monitoring generalization during training |
| Test | Final held-out evaluation |

The dataset's labels, categories, and severity annotations are used to construct structured supervised fine-tuning targets.

## How it works

The project follows a supervised fine-tuning pipeline.

1. **Dataset loading:** Load the prompt-injection dataset using Hugging Face Datasets.
2. **Preprocessing:** Filter and prepare the examples containing input text, label, category, and severity.
3. **Conversational formatting:** Convert each example into a system-user-assistant conversation using Qwen3's native chat template.
4. **Quantization:** Load the base model in 4-bit NF4 precision to reduce memory requirements.
5. **LoRA configuration:** Attach trainable low-rank adapters to selected model modules while keeping the quantized base weights frozen.
6. **Fine-tuning:** Train the adapters on the formatted dataset.
7. **Validation:** Monitor training and validation loss throughout training.
8. **Evaluation:** Generate predictions on the held-out test set and compare them against ground-truth annotations.
9. **Error analysis:** Inspect incorrect predictions and evaluate classification performance.

## QLoRA configuration

QLoRA combines quantization with Low-Rank Adaptation to make fine-tuning large language models more memory-efficient.

The base model is quantized and kept frozen. Instead of updating all model parameters, training updates small LoRA matrices attached to selected linear layers.

| Configuration | Value |
|---|---|
| Quantization type | NF4 |
| Quantization bits | 4 |
| Double quantization | Enabled |
| Compute dtype | FP16 |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Optimizer | Paged AdamW 8-bit |
| Learning rate | 2e-4 |
| Training steps | 500 |
| Effective batch size | 16 |

Only approximately **0.0533% of the model's parameters** were trainable during fine-tuning.

This substantially reduces the number of parameters requiring gradient updates compared with full fine-tuning.

## Evaluation methodology

Training loss and validation loss alone do not establish whether a model is an effective security classifier.

The final evaluation therefore uses generated predictions on the held-out test set.

### Classification metrics

- **Accuracy:** Overall proportion of correctly classified examples.
- **Precision:** Proportion of predicted malicious examples that are actually malicious.
- **Recall:** Proportion of actual malicious examples successfully detected.
- **Macro F1:** Unweighted average F1 across classes, giving each class equal importance.
- **Confusion matrix:** Breakdown of correct and incorrect predictions across classes.
- **False positive rate:** Proportion of benign examples incorrectly flagged as malicious.
- **False negative rate:** Proportion of malicious examples incorrectly classified as benign.

### Structured output evaluation

In addition to the primary benign/malicious classification task, the model is evaluated on:

- Attack category prediction.
- Severity prediction.

Macro F1 is used to assess performance across the category and severity classes.

### Error analysis

Incorrect predictions are retained for further inspection. This helps identify:

- Missed malicious inputs.
- Benign inputs incorrectly flagged as attacks.
- Incorrect attack-category predictions.
- Severity misclassification.

For security-oriented classification, false negatives are particularly important because they represent malicious inputs that the detector fails to identify.

## Training results

The training run completed 500 optimization steps in approximately 25 minutes.

| Step | Training Loss | Validation Loss | Mean Token Accuracy |
|---:|---:|---:|---:|
| 100 | 1.7623 | 1.4768 | 77.18% |
| 200 | 1.6595 | 1.3916 | 78.10% |
| 300 | 1.5948 | 1.3329 | 78.78% |
| 400 | 1.4775 | 1.3032 | 79.11% |
| 500 | 1.5436 | 1.2916 | 79.28% |

Validation loss decreased throughout the run, suggesting that the model continued to improve on the validation split.

The final training loss was `1.5436`, while the final validation loss was `1.2916`.

## Tech stack

- **Python** — Core implementation.
- **PyTorch** — Model training and tensor operations.
- **Hugging Face Transformers** — Model loading, tokenization, and generation.
- **Hugging Face Datasets** — Dataset loading and preprocessing.
- **PEFT** — Parameter-efficient fine-tuning with LoRA.
- **bitsandbytes** — 4-bit quantization and memory-efficient optimization.
- **TRL** — Supervised fine-tuning workflow.
- **scikit-learn** — Classification metrics and confusion matrix.
- **Matplotlib** — Evaluation visualizations.

## Repository structure

```text
.
├── FineTunedQwenThreatClassifierV1.ipynb
├── assets/
│   ├── training-results.png
│   └── confusion-matrix.png
├── README.md
└── ...
```

*The structure above highlights the primary training notebook and result artifacts. Additional files can be added as the project evolves.*

## Running the project

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support
- A compatible NVIDIA GPU
- Hugging Face account (for model and dataset access)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Qwen3-Prompt-Injection-Detector.git

cd Qwen3-Prompt-Injection-Detector
```

### 2. Install dependencies

```bash
pip install -U transformers datasets accelerate peft
pip install -U bitsandbytes trl
pip install -U scikit-learn matplotlib pandas
```

### 3. Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "neuralchemy/Prompt-injection-dataset",
    "full"
)

train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]
```

### 4. Run the training notebook

Open `FineTunedQwenThreatClassifierV1.ipynb` and execute the notebook cells in sequence.

The notebook covers:

- Dataset preparation.
- Qwen3 chat-template formatting.
- 4-bit model loading.
- LoRA configuration.
- Supervised fine-tuning.
- Validation monitoring.
- Test-set prediction and evaluation.

## Limitations and future improvements

This project is an experimental prompt-injection classifier, not a complete LLM application security solution.

Potential improvements include:

- Evaluate generalization on independently sourced prompt-injection datasets.
- Test against previously unseen attack categories and adversarially modified prompts.
- Compare different LoRA ranks and target-module configurations.
- Compare the fine-tuned model against the original base model using an identical evaluation protocol.
- Evaluate inference latency, throughput, and GPU memory consumption.
- Explore confidence calibration and threshold-based classification.
- Integrate the classifier into an LLM application as an additional input-screening layer.

A standalone classifier should not be treated as a substitute for robust system-prompt design, tool permissions, input validation, and application-level security controls.

## Acknowledgements

- [Qwen](https://huggingface.co/Qwen) for the Qwen3 model family.
- [Hugging Face](https://huggingface.co/) for the Transformers, Datasets, and PEFT ecosystem.
- [neuralchemy](https://huggingface.co/neuralchemy) for the Prompt Injection Dataset.

---

**Built as an exploration of parameter-efficient fine-tuning and LLM application security.**
