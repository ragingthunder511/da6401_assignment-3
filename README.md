# DA6401 Assignment 03 - Transliteration from English to Hindi Using Vanilla and Attention-Based Seq2Seq Models

## Overview

This project presents a character-level transliteration system for converting words between English (Latin script) and Hindi (Devanagari script). The system is built using two architectures: a standard sequence-to-sequence model and a variant that incorporates an attention mechanism to enhance performance and interpretability.

## Highlights

- **Dual Modeling Approach**: Implements both a baseline seq2seq model and an attention-enhanced version.
- **Attention Visualization**: Displays heatmaps to understand how the model aligns input and output characters.
- **Experiment Tracking**: Integrates with Weights & Biases (WandB) for systematic hyperparameter tuning.
- **Reusability**: Includes scripts and notebooks that allow for model training, evaluation, and prediction generation.
- **Performance Comparison**: Evaluates and contrasts the effectiveness of attention-based models versus vanilla approaches.

## Model Descriptions

### 1. Basic Seq2Seq Model
- Uses LSTM/GRU layers for both encoder and decoder.
- Relies on a fixed-size context vector to pass information from input to output.
- Suitable as a baseline for evaluating improvements from attention mechanisms.

### 2. Seq2Seq with Attention
- Implements Bahdanau-style additive attention.
- Dynamically computes a context vector for each decoding step based on encoder outputs.
- Improves alignment and performance, especially for longer sequences.

## Dataset

The model is trained and evaluated using the [Dakshina Dataset](https://github.com/google-research-datasets/dakshina), specifically focusing on Hindi transliteration.

| File Name                     | Number of Samples | Description      |
|------------------------------|-------------------|------------------|
| `hi.translit.sampled.train.tsv` | around 10,000            | Training Set     |
| `hi.translit.sampled.dev.tsv`   | around 1,000             | Validation Set   |
| `hi.translit.sampled.test.tsv`  | around 1,000             | Test Set         |

## Directory Layout

```
transliteration-project/
├── notebooks/
│ └── cs24m020_vanilla.ipynb # All-in-one notebook for training, evaluation, and analysis
├── predictions_attention/
│ └── output.csv # Format: input,prediction,ground-truth
├── predictions_vanilla/
│ └── output.csv # Format: input,prediction,ground-truth
├── attention_train.py # Script for training the attention-based model
├── vanilla_train.py # Script for training the vanilla seq2seq model
└── README.md # Project documentation
```



## Getting Started

### Prerequisites

Ensure the following Python packages are installed:

```bash
pip install torch numpy pandas tqdm matplotlib wandb
```

### Training Instructions

Use the following commands to initiate training for each model:

```bash
# Train the vanilla sequence-to-sequence model
python vanilla_train.py
```
```bash
# Train the attention-based sequence-to-sequence model
python attention_train.py
```

## Hyperparameter Sweep Configuration

The following hyperparameters were used for performing extensive sweeps to optimize the sequence-to-sequence transliteration model:

### 📦 Model & Training Parameters
- **batch_size**: `[16, 32, 64, 128, 256]`
- **num_epochs**: `[10]`
- **encoder_layers**: `[1, 2, 3]`
- **decoder_layers**: `[1, 2, 3]`
- **hidden_size**: `[16, 32, 64, 128, 256, 512, 1024]`
- **embedding_dim**: `[16, 32, 64, 256, 512]`
- **dropout_rate**: `[0.2, 0.3, 0.4]`
- **bi_directional**: `[True, False]`

### 🔍 Search & Decoding Parameters
- **beam_width**: `[1, 3, 5]`
- **teacher_forcing_ratio**: `[0.0, 0.3, 0.5, 0.7, 1.0]`
- **length_penalty**: `[0, 0.4, 0.5, 0.6]`

### ⚙️ Optimization Parameters
- **optimizer**: `["adam", "sgd", "rmsprop", "adagrad"]`
- **learning_rate**: `[0.005, 0.001, 0.01, 0.1]`

### 🔁 RNN Cell Variants
- **rnn_cell**: `["RNN", "GRU", "LSTM"]`

These combinations were used in a sweep setup to explore the model's performance across a broad spectrum of configurations. The goal was to identify optimal settings for training a robust and generalizable transliteration system.


These scripts are configured to automatically begin hyperparameter sweeps using Weights & Biases (WandB).
Before running, make sure to insert your WandB API key in the designated section of the scripts.
Training metrics, validation accuracy, and model behavior will be tracked and visualized on your WandB dashboard.

Author : Grishma Uday Karekar
Roll No : CS24M020
