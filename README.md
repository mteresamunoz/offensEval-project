# Offensive Language Detection in Social Media

This repository contains the implementation and experiments for detecting offensive language in Twitter data, based on the SemEval-2019 Task 6 (OffensEval) shared task.

## Project Overview

This project addresses the binary classification task of identifying whether tweets contain offensive language or not. The work explores multiple modeling approaches ranging from classical machine learning baselines to state-of-the-art transformer models.

**Task**: Sub-task A - Offensive Language Identification (Binary Classification)
- **OFF**: Offensive tweets (insults, threats, profanity)
- **NOT**: Non-offensive tweets

## Research Question

**How does Twitter-specific preprocessing affect offensive language detection performance across different model architectures?**

### Motivation

Social media text contains platform-specific features (hashtags, mentions, URLs, emojis) that may carry semantic information relevant to offense detection. However, traditional NLP pipelines often remove these elements as "noise." This project investigates whether such preprocessing helps or hurts different model types.

### Experimental Setup

I compare three preprocessing strategies:

1. **Raw**: No preprocessing - original tweet text as-is
2. **Clean**: Remove URLs and user mentions (@USER)
3. **Aggressive**: Remove URLs, mentions, hashtags, numbers, and special characters

Each strategy is evaluated across three model families:
- **Classical**: N-gram features with Naive Bayes and SVM
- **Neural**: LSTM with pre-trained embeddings (GloVe, FastText)
- **Transformers**: BERT, RoBERTa, and DeBERTa

### Hypothesis

- Classical models may benefit from cleaned text (less noise, clearer n-gram patterns)
- Transformer models, pre-trained on noisy social media data, may perform better with raw text
- Twitter-specific features (hashtags, mentions) could contain valuable signals for offensive content detection

Results and analysis are presented in the final report.


## Dataset

The dataset is derived from the **Offensive Language Identification Dataset (OLID)** introduced in SemEval-2019 Task 6 ([Zampieri et al., 2019](https://aclanthology.org/S19-2010/)).

### Dataset Statistics

| Split | Total Examples | OFF | NOT | OFF Proportion |
|-------|---------------|-----|-----|----------------|
| Train | 12,240 | 4,048 | 8,192 | 33.07% |
| Dev | 1,000 | 352 | 648 | 35.20% |
| Test | 860 | 240 | 620 | 27.91% |

**Key Characteristics**:
- Class imbalance: ~2:1 ratio (NOT:OFF)
- Average tweet length: ~125-146 characters
- Noisy text with Twitter-specific features (@mentions, URLs, hashtags)
- Real-world social media content with diverse offensive language patterns

### Data Format

Each TSV file contains two tab-separated columns:
    \<tweet_text>\t\<label>

### Data Preprocessing

To generate the preprocessed datasets for experiments:

```
# Generate all preprocessing variants
python scripts/preprocess.py --input_dir data/preprocessed/raw/ --output_dir data/preprocessed/ --strategy all

```

This creates three preprocessing variants:
- **raw/**: Original data without modifications
- **clean/**: Removes URLs and @mentions
- **aggressive/**: Removes URLs, mentions, hashtags, numbers, and special characters

Each variant maintains the same TSV format with tweet and label columns.

## Evaluation Metrics

Following the official SemEval-2019 Task 6 guidelines, I use:

- **Macro-averaged F1-score** (primary metric)
- Precision and Recall (per class and macro-averaged)
- Accuracy (for reference)

The macro F1-score is prioritized due to class imbalance, ensuring equal importance for both classes.

## Project Structure
offensEval-project/

    ├── baseline/ # Classical ML models (n-grams + NB/SVM)
    ├── deep/ # Deep learning models (LSTM, RNN)
    ├── transformers/ # Transformer-based models (BERT, RoBERTa, DeBERTa)
    ├── scripts/ # Utility scripts (preprocessing, data exploration)
    ├── data/ # Dataset files 
        ├── preprocessed/
        │   ├── raw/          # Original data (no changes)
        │   │   ├── train.tsv
        │   │   ├── dev.tsv
        │   │   └── test.tsv
        │   ├── clean/        # URLs and mentions removed
        │   │   ├── train.tsv
        │   │   ├── dev.tsv
        │   │   └── test.tsv
        │   └── aggressive/   # Heavy preprocessing
        │       ├── train.tsv
        │       ├── dev.tsv
        │       └── test.tsv
    ├── results/ # Experiment results and metrics
    ├── docs/ # Documentation and report
    └── README.md # This file


## Models Implemented

### 1. Classical Baselines
- N-gram features (unigrams + bigrams) with TF-IDF vectorization
- Multinomial Naive Bayes
- Support Vector Machine (SVM)

### 2. Neural Networks
- LSTM with static embeddings (GloVe, FastText)
- Bidirectional LSTM variants

### 3. Transformer Models
- BERT (bert-base-uncased)
- RoBERTa (roberta-base)
- DeBERTa (microsoft/deberta-v3-base)

## Installation

```bash

# Clone the repository
git clone https://github.com/mteresamunoz/offensEval-project.git 
cd offensEval-project

# Create virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Exploration
```
python scripts/explore_data.py --input data/preprocessed/raw/train.tsv --output results/train_stats.txt
```

### Preprocessing Data

```
# Generate all preprocessing variants
python scripts/preprocess.py --input_dir data/preprocessed/raw/ --output_dir data/preprocessed/ --strategy all
```

### Training Models

**Baseline models:**
```
python baseline/ngram_classifier.py --train data/train.tsv --dev data/dev.tsv --test data/test.tsv --model svm --output results/
```

**LSTM models:**
```
python deep/lstm_classifier.py --train data/train.tsv --dev data/dev.tsv --test data/test.tsv --embedding glove --output results/
```

**Transformer models:**
```
python transformers/transformer_classifier.py --train data/train.tsv --dev data/dev.tsv --test data/test.tsv --model bert-base-uncased --output results/
```

## Results

Full experimental results and analysis are documented in the final report (see `docs/`).

## References

- Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., & Kumar, R. (2019). SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval). *Proceedings of SemEval*.

## License

This project is developed for academic purposes as part of the Learning from Data course.

## Contact

María Teresa Muñoz Martín - m.t.munoz.martin@student.rug.nl

Project Link: [https://github.com/mteresamunoz/offensEval-project](https://github.com/mteresamunoz/offensEval-project)
