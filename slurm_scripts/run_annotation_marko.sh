#!/bin/bash
#SBATCH --job-name=qwen_annotation
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --output=/home/s3758869/synchain-absa-emotion/slurm_outputs/qwen_annotation_%j.out
#SBATCH --error=/home/s3758869/synchain-absa-emotion/slurm_outputs/qwen_annotation_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

source /home/s3758869/absa_synchain/bin/activate
cd /home/s3758869/synchain-absa-emotion
python scripts/annotation/annotate.py