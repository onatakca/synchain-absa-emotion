#!/bin/bash
#SBATCH --job-name=knoledge_distillation
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --mem=40GB
#SBATCH --output=/home/s3758869/synchain-absa-emotion/slurm_outputs/kd%j.out

CONFIG_PATH="/home/s3758869/synchain-absa-emotion/scripts/modeling/configs/meta8b/evaluation_meta8b.json"
source /home/s3758869/absa_synchain/bin/activate
cd /home/s3758869/synchain-absa-emotion
export PYTHONPATH=/home/s3758869/synchain-absa-emotion:$PYTHONPATH
python scripts/modeling/knowledge_distillation.py "$CONFIG_PATH"
