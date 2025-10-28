Write-Host "======================================"
Write-Host "LSTM EXPERIMENTS - OFFENSIVE LANGUAGE DETECTION"
Write-Host "======================================"
Write-Host ""

# Create results directory if it doesn't exist
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "Starting GloVe experiments..."
Write-Host ""

# GloVe experiments
Write-Host "[1/6] Running: GloVe + Raw"
python deep/lstm_classifier.py --train data/preprocessed/raw/train.tsv --dev data/preprocessed/raw/dev.tsv --test data/preprocessed/raw/test.tsv --embeddings embeddings/glove.twitter.27B.200d.txt --embedding_type glove --embedding_dim 200 --preprocessing raw --output results/

Write-Host ""
Write-Host "[2/6] Running: GloVe + Clean"
python deep/lstm_classifier.py --train data/preprocessed/clean/train.tsv --dev data/preprocessed/clean/dev.tsv --test data/preprocessed/clean/test.tsv --embeddings embeddings/glove.twitter.27B.200d.txt --embedding_type glove --embedding_dim 200 --preprocessing clean --output results/

Write-Host ""
Write-Host "[3/6] Running: GloVe + Aggressive"
python deep/lstm_classifier.py --train data/preprocessed/aggressive/train.tsv --dev data/preprocessed/aggressive/dev.tsv --test data/preprocessed/aggressive/test.tsv --embeddings embeddings/glove.twitter.27B.200d.txt --embedding_type glove --embedding_dim 200 --preprocessing aggressive --output results/

Write-Host ""
Write-Host "Starting FastText experiments..."
Write-Host ""

# FastText experiments
Write-Host "[4/6] Running: FastText + Raw"
python deep/lstm_classifier.py --train data/preprocessed/raw/train.tsv --dev data/preprocessed/raw/dev.tsv --test data/preprocessed/raw/test.tsv --embeddings embeddings/fasttext-wiki-news-300d.txt --embedding_type fasttext --embedding_dim 300 --preprocessing raw --output results/

Write-Host ""
Write-Host "[5/6] Running: FastText + Clean"
python deep/lstm_classifier.py --train data/preprocessed/clean/train.tsv --dev data/preprocessed/clean/dev.tsv --test data/preprocessed/clean/test.tsv --embeddings embeddings/fasttext-wiki-news-300d.txt --embedding_type fasttext --embedding_dim 300 --preprocessing clean --output results/

Write-Host ""
Write-Host "[6/6] Running: FastText + Aggressive"
python deep/lstm_classifier.py --train data/preprocessed/aggressive/train.tsv --dev data/preprocessed/aggressive/dev.tsv --test data/preprocessed/aggressive/test.tsv --embeddings embeddings/fasttext-wiki-news-300d.txt --embedding_type fasttext --embedding_dim 300 --preprocessing aggressive --output results/

Write-Host ""
Write-Host "======================================"
Write-Host "ALL LSTM EXPERIMENTS COMPLETED"
Write-Host "======================================"
Write-Host ""

Write-Host "Results CSVs:"
Get-ChildItem results/lstm_*.csv | Format-Table Name, Length

Write-Host ""
Write-Host "Error analysis reports:"
Get-ChildItem results/error_analysis_bilstm*.txt | Format-Table Name

Write-Host ""
Write-Host "Predictions files:"
Get-ChildItem results/predictions_bilstm*.csv | Format-Table Name

Write-Host ""
Write-Host "Confusion matrices:"
Get-ChildItem results/confusion_matrix_bilstm*.png | Format-Table Name

Write-Host ""
Write-Host "To visualize results, run:"
Write-Host "  python scripts/visualize_results.py --pattern lstm"
