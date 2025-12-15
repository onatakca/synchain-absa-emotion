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
mkdir -p /home/s2457997/synchain-absa-emotion/hf_cache

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

# Prefer a local model path if present to avoid network
for CAND in \
	"/home/s2457997/synchain-absa-emotion/models/Qwen2.5-7B-Instruct" \
	"/home/s2457997/synchain-absa-emotion/models/Qwen1.5-7B-Chat"; do
	if [ -d "$CAND" ]; then
		export QWEN_LOCAL_PATH="$CAND"
		echo "Using local model at: $QWEN_LOCAL_PATH"
		break
	fi
done

# Set HF caches inside repo to keep artifacts contained
export HF_HOME="/home/s2457997/synchain-absa-emotion/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
echo "QWEN_LOCAL_PATH=${QWEN_LOCAL_PATH:-<none>}"
echo "HF_HOME=$HF_HOME"

# Be nice to shared node: limit torch threads
"$PYTHON" -c "import torch; torch.set_num_threads(max(1, torch.get_num_threads()//2)); print('Threads:', torch.get_num_threads())" || true

# Run preprocessing (emoji cleanup + CUDA-only LLM categorization with progress + incremental saves)
"$PYTHON" /home/s2457997/synchain-absa-emotion/scripts/preprocessing/preprocess.py

# Tail result location
echo "Done. Categorised outputs are in: /home/s2457997/synchain-absa-emotion/data/output_data"
echo "Examples:"
ls -1 /home/s2457997/synchain-absa-emotion/data/output_data/*_categorised.csv 2>/dev/null || echo "(No categorised files yet)"