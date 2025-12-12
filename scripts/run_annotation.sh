#!/bin/bash
#SBATCH --job-name=qwen_test50
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --output=annotation_conversational_%j.log

echo "=== QWEN TEACHER ANNOTATION - CONVERSATIONAL TWEETS ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Change to project directory
cd /home/s3521281/synchain-absa-emotion || exit 1
echo "Working directory: $(pwd)"

# Add project to Python path
export PYTHONPATH="/home/s3521281/synchain-absa-emotion:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Disable online checks (compute nodes have limited internet)
echo "Using local cached models only"
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run the Python annotation script (conversational filter)
echo "Running annotation (conversational tweets only)..."
python3 scripts/annotate.py

echo ""
echo "Finished: $(date)"
echo "=== JOB COMPLETE ==="
