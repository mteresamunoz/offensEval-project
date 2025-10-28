# scripts/run_baselines.ps1
# Baseline Experiments for Offensive Language Detection

Write-Host "======================================"
Write-Host "BASELINE EXPERIMENTS - OFFENSIVE LANGUAGE DETECTION"
Write-Host "======================================"
Write-Host ""

# Create results directory if it doesn't exist
New-Item -ItemType Directory -Force -Path results | Out-Null

Write-Host "Starting SVM experiments..."
Write-Host ""

# SVM experiments (3 preprocessing strategies)
Write-Host "[1/6] Running: SVM + Raw"
python baseline/ngram_classifier.py `
    --train data/preprocessed/raw/train.tsv `
    --dev data/preprocessed/raw/dev.tsv `
    --test data/preprocessed/raw/test.tsv `
    --model svm `
    --preprocessing raw `
    --output results/

Write-Host ""
Write-Host "[2/6] Running: SVM + Clean"
python baseline/ngram_classifier.py `
    --train data/preprocessed/clean/train.tsv `
    --dev data/preprocessed/clean/dev.tsv `
    --test data/preprocessed/clean/test.tsv `
    --model svm `
    --preprocessing clean `
    --output results/

Write-Host ""
Write-Host "[3/6] Running: SVM + Aggressive"
python baseline/ngram_classifier.py `
    --train data/preprocessed/aggressive/train.tsv `
    --dev data/preprocessed/aggressive/dev.tsv `
    --test data/preprocessed/aggressive/test.tsv `
    --model svm `
    --preprocessing aggressive `
    --output results/

Write-Host ""
Write-Host "Starting Naive Bayes experiments..."
Write-Host ""

# Naive Bayes experiments (3 preprocessing strategies)
Write-Host "[4/6] Running: Naive Bayes + Raw"
python baseline/ngram_classifier.py `
    --train data/preprocessed/raw/train.tsv `
    --dev data/preprocessed/raw/dev.tsv `
    --test data/preprocessed/raw/test.tsv `
    --model nb `
    --preprocessing raw `
    --output results/

Write-Host ""
Write-Host "[5/6] Running: Naive Bayes + Clean"
python baseline/ngram_classifier.py `
    --train data/preprocessed/clean/train.tsv `
    --dev data/preprocessed/clean/dev.tsv `
    --test data/preprocessed/clean/test.tsv `
    --model nb `
    --preprocessing clean `
    --output results/

Write-Host ""
Write-Host "[6/6] Running: Naive Bayes + Aggressive"
python baseline/ngram_classifier.py `
    --train data/preprocessed/aggressive/train.tsv `
    --dev data/preprocessed/aggressive/dev.tsv `
    --test data/preprocessed/aggressive/test.tsv `
    --model nb `
    --preprocessing aggressive `
    --output results/

Write-Host ""
Write-Host "======================================"
Write-Host "ALL BASELINE EXPERIMENTS COMPLETED"
Write-Host "======================================"
Write-Host ""
Write-Host "Results saved in results/ directory:"
Get-ChildItem results/baseline_*.csv | Format-Table Name, Length

Write-Host ""
Write-Host "Error analysis files:"
Get-ChildItem results/error_analysis_*.txt | Format-Table Name

Write-Host ""
Write-Host "Confusion matrices:"
Get-ChildItem results/confusion_matrix_*.png | Format-Table Name

Write-Host ""
Write-Host "To visualize results, run:"
Write-Host "  python scripts/visualize_results.py --results_dir results/ --pattern baseline"
