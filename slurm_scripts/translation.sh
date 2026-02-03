#!/bin/bash
#SBATCH --job-name=ch_to_eng_translation
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --mem=40GB
#SBATCH --output=/home/s3758869/synchain-absa-emotion/slurm_outputs/ch_to_eng_translation%j.out

source /home/s3758869/absa_synchain/bin/activate
cd /home/s3758869/synchain-absa-emotion
export PYTHONPATH=/home/s3758869/synchain-absa-emotion:$PYTHONPATH
python scripts/minsi_data/translation.py
