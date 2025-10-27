#!/bin/bash

echo "======================================"
echo "LSTM EXPERIMENTS - OFFENSIVE LANGUAGE DETECTION"
echo "======================================"

# Create results directory if it doesn't exist
mkdir -p results/

echo ""
echo "Starting GloVe experiments..."
echo ""

# GloVe experiments
echo "[1/6] Running: GloVe + Raw"
python deep/lstm_classifier.py \
    --train data/preprocessed/raw/train.tsv \
    --dev data/preprocessed/raw/dev.tsv \
    --test data/preprocessed/raw/test.tsv \
    --embeddings embeddings/glove.twitter.27B.200d.txt \
    --embedding_type glove \
    --embedding_dim 200 \
    --preprocessing raw \
    --output results/

echo ""
echo "[2/6] Running: GloVe + Clean"
python deep/lstm_classifier.py \
    --train data/preprocessed/clean/train.tsv \
    --dev data/preprocessed/clean/dev.tsv \
    --test data/preprocessed/clean/test.tsv \
    --embeddings embeddings/glove.twitter.27B.200d.txt \
    --embedding_type glove \
    --embedding_dim 200 \
    --preprocessing clean \
    --output results/

echo ""
echo "[3/6] Running: GloVe + Aggressive"
python deep/lstm_classifier.py \
    --train data/preprocessed/aggressive/train.tsv \
    --dev data/preprocessed/aggressive/dev.tsv \
    --test data/preprocessed/aggressive/test.tsv \
    --embeddings embeddings/glove.twitter.27B.200d.txt \
    --embedding_type glove \
    --embedding_dim 200 \
    --preprocessing aggressive \
    --output results/

echo ""
echo "Starting FastText experiments..."
echo ""

# FastText experiments
echo "[4/6] Running: FastText + Raw"
python deep/lstm_classifier.py \
    --train data/preprocessed/raw/train.tsv \
    --dev data/preprocessed/raw/dev.tsv \
    --test data/preprocessed/raw/test.tsv \
    --embeddings embeddings/fasttext-wiki-news-300d.txt \
    --embedding_type fasttext \
    --embedding_dim 300 \
    --preprocessing raw \
    --output results/

echo ""
echo "[5/6] Running: FastText + Clean"
python deep/lstm_classifier.py \
    --train data/preprocessed/clean/train.tsv \
    --dev data/preprocessed/clean/dev.tsv \
    --test data/preprocessed/clean/test.tsv \
    --embeddings embeddings/fasttext-wiki-news-300d.txt \
    --embedding_type fasttext \
    --embedding_dim 300 \
    --preprocessing clean \
    --output results/

echo ""
echo "[6/6] Running: FastText + Aggressive"
python deep/lstm_classifier.py \
    --train data/preprocessed/aggressive/train.tsv \
    --dev data/preprocessed/aggressive/dev.tsv \
    --test data/preprocessed/aggressive/test.tsv \
    --embeddings embeddings/fasttext-wiki-news-300d.txt \
    --embedding_type fasttext \
    --embedding_dim 300 \
    --preprocessing aggressive \
    --output results/

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "======================================"
echo ""
echo "Results saved in results/ directory:"
ls -lh results/lstm_*.csv

echo ""
echo "To visualize results, run:"
echo "python scripts/visualize_results.py"
