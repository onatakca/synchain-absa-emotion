#!/bin/bash
#SBATCH --job-name=preprocess_news
#SBATCH --output=/home/s3758869/synchain-absa-emotion/logs/preprocess_news_%j.out
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00

mkdir -p /home/s3758869/synchain-absa-emotion/logs

export PYTHONPATH="/home/s3758869/synchain-absa-emotion:$PYTHONPATH"

cd /home/s3758869/synchain-absa-emotion
python3 scripts/preprocessing/preprocess.py	