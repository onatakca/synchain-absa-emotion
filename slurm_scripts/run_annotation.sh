#!/bin/bash
#SBATCH --job-name=qwen_test50
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --output=annotation_conversational_%j.log

echo "Job ID: $SLURM_JOB_ID"

cd /home/s3758869/synchain-absa-emotion 

export PYTHONPATH="/home/s3758869/synchain-absa-emotion:$PYTHONPATH"

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python3 scripts/annotation/annotate.py