#!/bin/bash

echo "======================================"
echo "TRANSFORMER EXPERIMENTS - OFFENSIVE LANGUAGE DETECTION"
echo "======================================"
echo ""

# Create results directory
mkdir -p results

# Check GPU
echo "Checking GPU..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# BERT experiments
echo "======================================"
echo "BERT-BASE EXPERIMENTS"
echo "======================================"

echo "[1/6] BERT + Raw"
python transformers/transformer_classifier.py \
    --train data/preprocessed/raw/train.tsv \
    --dev data/preprocessed/raw/dev.tsv \
    --test data/preprocessed/raw/test.tsv \
    --model_name bert-base-uncased \
    --preprocessing raw \
    --batch_size 16 \
    --output results/

echo ""
echo "[2/6] BERT + Clean"
python transformers/transformer_classifier.py \
    --train data/preprocessed/clean/train.tsv \
    --dev data/preprocessed/clean/dev.tsv \
    --test data/preprocessed/clean/test.tsv \
    --model_name bert-base-uncased \
    --preprocessing clean \
    --batch_size 16 \
    --output results/

echo ""
echo "[3/6] BERT + Aggressive"
python transformers/transformer_classifier.py \
    --train data/preprocessed/aggressive/train.tsv \
    --dev data/preprocessed/aggressive/dev.tsv \
    --test data/preprocessed/aggressive/test.tsv \
    --model_name bert-base-uncased \
    --preprocessing aggressive \
    --batch_size 16 \
    --output results/

# RoBERTa experiments
echo ""
echo "======================================"
echo "ROBERTA-BASE EXPERIMENTS"
echo "======================================"

echo "[4/6] RoBERTa + Raw"
python transformers/transformer_classifier.py \
    --train data/preprocessed/raw/train.tsv \
    --dev data/preprocessed/raw/dev.tsv \
    --test data/preprocessed/raw/test.tsv \
    --model_name roberta-base \
    --preprocessing raw \
    --batch_size 16 \
    --output results/

echo ""
echo "[5/6] RoBERTa + Clean"
python transformers/transformer_classifier.py \
    --train data/preprocessed/clean/train.tsv \
    --dev data/preprocessed/clean/dev.tsv \
    --test data/preprocessed/clean/test.tsv \
    --model_name roberta-base \
    --preprocessing clean \
    --batch_size 16 \
    --output results/

echo ""
echo "[6/6] RoBERTa + Aggressive"
python transformers/transformer_classifier.py \
    --train data/preprocessed/aggressive/train.tsv \
    --dev data/preprocessed/aggressive/dev.tsv \
    --test data/preprocessed/aggressive/test.tsv \
    --model_name roberta-base \
    --preprocessing aggressive \
    --batch_size 16 \
    --output results/

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "======================================"
echo ""
echo "Results saved in results/"
ls -lh results/transformer_*.csv

echo ""
echo "✅ DONE!"
