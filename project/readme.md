# Multilingual Translation Model

This project implements a BPE-based trilingual tokenizer and encoder-decoder model for translation tasks. The system supports translation from multiple source languages (default: Swedish and Italian) to English.

## Table of Contents

- [Components Overview](#components-overview)
- [Setup](#setup)
- [Usage](#usage)
  - [Creating a Tokenizer](#creating-a-tokenizer)
  - [Tokenizing the Dataset](#tokenizing-the-dataset)
  - [Training a Model](#training-a-model)
  - [Generating Translations](#generating-translations)
  - [Evaluating the Model](#evaluating-the-model)
  - [Translating Examples](#translating-examples)
  - [Counting Model Parameters](#counting-model-parameters)

---

## Components Overview

### Tokenizer
Creates a BPE trilingual tokenizer with a specified vocabulary size. The tokenizer balances samples across two source languages (`lang_1`, `lang_2`) and the target language (English).

### Data Preprocessing
Loads and formats data with configurable size limits. Source languages are specified via `--l1` and `--l2` parameters (default: Swedish and Italian).

### Model
Contains the encoder-decoder model architecture.

### Trainer
Handles model training and evaluation after each epoch.

### Translate
Implements the generation logic for producing translations.

---

## Setup

Create a virtual environment and install dependencies:

```bash
# Create the python virtual environment
python -m venv <name_of_venv>

# Activate the virtual environment
source <name_of_venv>/bin/activate

# Install required packages
pip install -r requirements.txt
```

Note that you need to change the venv that gets activated in `run.sh` to the name of your venv

---

## Usage

### Creating a Tokenizer

Creates a new BPE tokenizer. This is CPU-intensive and does not require a GPU.

**Note:** If you have a pre-trained model, download its tokenizer instead of creating a new one.

**Command:**
```bash
python main.py --run="tokenizer" --data-limit=1000000
```

**Optional Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--l1` | `"sv"` | First source language to translate from |
| `--l2` | `"it"` | Second source language to translate from |
| `--vocab-size` | `50000` | Vocabulary size of the tokenizer |
| `--token-output-dir` | `"my_tokenizer"` | Directory where the tokenizer will be saved |
| `--data-limit` | `1000000` | Number of sequence pairs to load during tokenization |

---

### Tokenizing the Dataset

Pre-tokenizes datasets to avoid repeated tokenization during training or evaluation. Use the same dataset size as during tokenization.

**Command:**
```bash
python main.py --run="encode dataset" --data-limit=1000000 --token-ds-out-path="tokenized_datasets/"
```

**Optional Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--l1` | `"sv"` | First source language to translate from |
| `--l2` | `"it"` | Second source language to translate from |
| `--token-ds-out-path` | `"tokenized_datasets/"` | Directory where tokenized datasets will be saved |
| `--data-limit` | `1000000` | Number of sequence pairs to load during encoding |

---

### Training a Model

Trains the translation model. This is GPU-intensive and recommended to run on a computing cluster.

**Command:**
```bash
sbatch -p long run.sh --run="train" --dataset-load-name="sv_en_dataset_tokenized" --save-model-dir="my_sv_en_model"
```

You can specify additional arguments after `--run="train"` to override defaults.

**Training Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `5` | Number of training epochs |
| `--lr` | `0.0001` | Learning rate |
| `--batch-size` | `32` | Batch size for training |
| `--save-model-dir` | `"trained_model"` | Directory to save the trained model |
| `--load-model-dir` | `None` | Directory to load a pre-trained model from |
| `--dataset-load-name` | `"sv_en_dataset_tokenized"` | Name of the tokenized dataset to use |

**Model Architecture Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-size` | `256` | Hidden layer size |
| `--intermediate-size` | `512` | Intermediate layer size |
| `--num-attention-heads` | `4` | Number of attention heads |
| `--num-hidden-layers` | `5` | Number of hidden layers |
| `--rms-norm-eps` | `0.001` | RMS normalization epsilon |

---

### Generating Translations

Interactive translation generation. **Note:** Cannot run on cluster environments as it uses `input()`. Note that if 
you use `--use-helsinki` the language of the model to load also has to be specified.

**Command:**
```bash
# Using a custom model
python main.py --run="gen" --load-model-dir="my_sv_en_model" --token-output-dir="my_tokenizer"

#Using a model from Helsinki University
python main.py --run="gen" --is-helsinki --helsinki-model-language="sv"
```

**Optional Arguments:**

| Argument | Description |
|----------|-------------|
| `--load-model-dir` | Directory of the model to load |
| `--token-output-dir` | Directory of the tokenizer |
| `--is-helsinki` | Add this if you want to load and run a model from Helsinki Uniersity |
| `--helsinki-model-language` | The language of the Helsinki model |

---

### Evaluating the Model

Evaluates model performance using standard metrics: BLEU, chrF, and COMET.

**Command:**
```bash
sbatch -p long run.sh --run="eval" --load-model-dir="my_sv_en_model" --dataset-load-name="sv_en_dataset_tokenized" --token-output-dir="my_tokenizer"
```

**Optional Arguments:**

| Argument | Description |
|----------|-------------|
| `--token-ds-out-path` | Path to the tokenized datasets folder |
| `--dataset-load-name` | Name of the dataset for evaluation |
| `--load-model-dir` | Directory of the model to load |
| `--token-output-dir` | Directory of the tokenizer used during training |

---

### Translating Examples

Evaluates and displays sample translations for manual inspection.

**Command:**
```bash
# Using sbatch
sbatch run.sh --run="translate examples" --load-model-dir="my_sv_en_model"  --token-output-dir="my_tokenizer"

# Or directly with Python
python main.py --run="translate examples" --load-model-dir="my_sv_en_model"  --token-output-dir="my_tokenizer"
```

**Optional Arguments:**

| Argument | Description |
|----------|-------------|
| `--load-model-dir` | Directory of the model to use |
| `--token-output-dir` | Directory of the tokenizer to use |

---

### Counting Model Parameters

Displays the total number of parameters for a given model configuration.

**Command:**
```bash
python main.py --run="params"
```

**Model Configuration Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-size` | `256` | Hidden layer size |
| `--intermediate-size` | `512` | Intermediate layer size |
| `--num-attention-heads` | `4` | Number of attention heads |
| `--num-hidden-layers` | `5` | Number of hidden layers |
| `--rms-norm-eps` | `0.001` | RMS normalization epsilon |

---

## Notes

- Always use the same tokenizer that was used during model training for evaluation and generation
- Ensure dataset size consistency between tokenization and encoding steps
- GPU resources are required for training and evaluation tasks