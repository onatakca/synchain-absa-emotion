#!/bin/bash
#SBATCH --job-name=preprocess_news
#SBATCH --output=/home/s2457997/synchain-absa-emotion/logs/preprocess_news_%j.out
#SBATCH --error=/home/s2457997/synchain-absa-emotion/logs/preprocess_news_%j.err
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00

# Optional: set account if required by your cluster
# #SBATCH --account=<your_account>

set -euo pipefail

# Create logs directory if missing
mkdir -p /home/s2457997/synchain-absa-emotion/logs

# Load modules if your cluster uses them (uncomment and adjust as needed)
# module purge
# module load cuda/12.1
# module load python/3.10

# Select Python: prefer user venv if present, else system Python
PYTHON="$HOME/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
	echo "[WARN] Venv python not found at $PYTHON; falling back to system python3"
	PYTHON="$(command -v python3 || command -v python)"
fi

# Show environment summary
nvidia-smi || true
"$PYTHON" -V
"$PYTHON" -c "import sys; print('Python exec:', sys.executable)"
"$PYTHON" -c "
try:
	import torch
	print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())
except Exception as e:
	print('Torch not available:', e)
"

# Be nice to shared node: limit torch threads
"$PYTHON" -c "import torch; torch.set_num_threads(max(1, torch.get_num_threads()//2)); print('Threads:', torch.get_num_threads())" || true

# Run preprocessing (emoji cleanup + CUDA-only LLM categorization with progress + incremental saves)
"$PYTHON" /home/s2457997/synchain-absa-emotion/scripts/preprocessing/preprocess.py

# Tail result location
echo "Done. Check output: /home/s2457997/synchain-absa-emotion/data/output_data/Corona_NLP_test_categorized.csv"