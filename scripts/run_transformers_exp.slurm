#!/usr/bin/env bash
#SBATCH --job-name=offenseval_deberta_bertweet
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
<<<<<<< HEAD
#SBATCH --time=24:00:00                   
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/offenseval-project/logs/transformers.out
#SBATCH --error=/gaueko1/users/mmartin/offenseval-project/logs/transformers.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mariateresa.munoz@ehu.eus
 

source /gaueko1/users/mmartin/offenseval-project/venv/bin/activate
=======
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/offensEval-project/logs/transformers_deberta_bertweet.out
#SBATCH --error=/gaueko1/users/mmartin/offensEval-project/logs/transformers_deberta_bertweet.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mariateresa.munoz@ehu.eus


# IMPORTANTE: Cd al directorio del proyecto
cd /gaueko1/users/mmartin/offensEval-project


# Activar entorno virtual
source venv/bin/activate
>>>>>>> bb690a0152d6fa4d65846c5d4ce5b6926081e5f9


echo "======================================"
echo "TRANSFORMER EXPERIMENTS - OFFENSIVE LANGUAGE DETECTION"
echo "DeBERTa-v3-base + BERTweet"
echo "======================================"
echo ""
<<<<<<< HEAD

=======
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""
>>>>>>> bb690a0152d6fa4d65846c5d4ce5b6926081e5f9


# Check GPU
echo "Checking GPU..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""


echo "======================================"
echo "EXPERIMENTS WITH DEBERTA-V3-BASE"
echo "======================================"


echo ""
echo "[1/6] DeBERTa-v3 + Raw"
python transformers/transformer_classifier.py \
    --train data/preprocessed/raw/train.tsv \
    --dev data/preprocessed/raw/dev.tsv \
    --test data/preprocessed/raw/test.tsv \
    --model_name microsoft/deberta-v3-base \
    --preprocessing raw \
    --batch_size 16 \
    --output results/


echo ""
echo "[2/6] DeBERTa-v3 + Clean"
python transformers/transformer_classifier.py \
    --train data/preprocessed/clean/train.tsv \
    --dev data/preprocessed/clean/dev.tsv \
    --test data/preprocessed/clean/test.tsv \
    --model_name microsoft/deberta-v3-base \
    --preprocessing clean \
    --batch_size 16 \
    --output results/


echo ""
echo "[3/6] DeBERTa-v3 + Aggressive"
python transformers/transformer_classifier.py \
    --train data/preprocessed/aggressive/train.tsv \
    --dev data/preprocessed/aggressive/dev.tsv \
    --test data/preprocessed/aggressive/test.tsv \
    --model_name microsoft/deberta-v3-base \
    --preprocessing aggressive \
    --batch_size 16 \
    --output results/


echo ""
echo "======================================"
echo "EXPERIMENTS WITH BERTWEET"
echo "======================================"


echo ""
echo "[4/6] BERTweet + Raw"
python transformers/transformer_classifier.py \
    --train data/preprocessed/raw/train.tsv \
    --dev data/preprocessed/raw/dev.tsv \
    --test data/preprocessed/raw/test.tsv \
    --model_name vinai/bertweet-base \
    --preprocessing raw \
    --batch_size 16 \
    --output results/


echo ""
echo "[5/6] BERTweet + Clean"
python transformers/transformer_classifier.py \
    --train data/preprocessed/clean/train.tsv \
    --dev data/preprocessed/clean/dev.tsv \
    --test data/preprocessed/clean/test.tsv \
    --model_name vinai/bertweet-base \
    --preprocessing clean \
    --batch_size 16 \
    --output results/


echo ""
echo "[6/6] BERTweet + Aggressive"
python transformers/transformer_classifier.py \
    --train data/preprocessed/aggressive/train.tsv \
    --dev data/preprocessed/aggressive/dev.tsv \
    --test data/preprocessed/aggressive/test.tsv \
    --model_name vinai/bertweet-base \
    --preprocessing aggressive \
    --batch_size 16 \
    --output results/


echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "======================================"
echo ""
echo "Results summary:"
echo ""
echo "DeBERTa-v3-base results:"
ls -lh results/transformer_deberta-v3-base*.csv
echo ""
echo "BERTweet results:"
ls -lh results/transformer_bertweet*.csv


echo ""
echo "✅ DONE!"
