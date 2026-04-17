# LLM Quantization on NLP Tasks

Research project exploring the impact of **dynamic INT8 quantization** on FLAN-T5-base across three NLP tasks: sentiment classification, math problem solving, and dialogue summarization.

## Overview

The project fine-tunes [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on each task, then applies `torch.quantization.quantize_dynamic` to compress the model and evaluates the trade-off between model size and task performance.

## Repository Structure

```
├── FLAN_T5_x_IMDB.ipynb                    # Fine-tune FLAN-T5 on IMDB sentiment
├── Quantization_FLAN_T5_Base_x_IMDB.ipynb  # Quantize and evaluate on IMDB
├── GSM8K_x_FLAN_T5.ipynb                   # Fine-tune FLAN-T5 on GSM8K math
├── Quantization_FLAN_T5_Base_x_GSM8K.ipynb # Quantize and evaluate on GSM8K
├── SAMsum_x_FLAN_T5_base.ipynb             # Fine-tune FLAN-T5 on SAMsum dialogue
├── Quantized_FLAN_T5_x_SAMsum.ipynb        # Quantize and evaluate on SAMsum
├── utils.py                                 # Shared utilities (seeds, checkpointing, metrics)
├── requirements.txt                         # Python dependencies
└── Dissertation Thesis.pdf                  # Full research write-up
```

## Tasks

| Task | Dataset | Metric | Notebook |
|------|---------|--------|----------|
| Sentiment Classification | [IMDB](https://huggingface.co/datasets/imdb) | Accuracy, Loss | `FLAN_T5_x_IMDB.ipynb` |
| Math Problem Solving | [GSM8K](https://huggingface.co/datasets/gsm8k) | Loss | `GSM8K_x_FLAN_T5.ipynb` |
| Dialogue Summarization | [SAMsum](https://huggingface.co/datasets/samsum) | ROUGE-1/2/L, Loss | `SAMsum_x_FLAN_T5_base.ipynb` |

## Quantization Results

| Task | Original Size | Quantized Size | Size Reduction | Performance Impact |
|------|--------------|----------------|----------------|-------------------|
| IMDB | ~990 MB | ~99 MB | ~90% | Accuracy: 94.66% → ~94.6% |
| GSM8K | ~990 MB | ~99 MB | ~90% | See dissertation for details |
| SAMsum | ~990 MB | ~99 MB | ~90% | See dissertation for details |

Quantization method: `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Google Colab

These notebooks are designed to run on Google Colab with GPU acceleration.

1. Mount Google Drive and set your model checkpoint directory:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   # Update this path to match your Drive structure
   DRIVE_BASE = '/content/drive/MyDrive/LLM Models'
   ```

2. Upload `utils.py` to your Colab session or copy it to Drive and import it.

3. Run the fine-tuning notebook for a task before running the corresponding quantization notebook — the quantization notebooks load checkpoints saved during fine-tuning.

## Training Configuration

| Hyperparameter | IMDB | GSM8K | SAMsum |
|---------------|------|-------|--------|
| Learning rate | 1e-5 | 5e-5 | 1e-4 |
| Batch size (train) | 32 | 8 | 16 |
| Epochs | 10 | 10 | 10 |
| Early stopping | No | patience=3 | No |
| Warmup | 10% of steps | 10% of steps | 10% of steps |
