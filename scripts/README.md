# Scripts Directory

Batch execution scripts for the Syn-Chain ABSA annotation pipeline.

## Annotation Scripts

### annotate.py
Generate reasoning traces for **conversational tweets only** using Qwen2.5-72B teacher model.

**Features:**
- Automatically filters out news-like tweets
- Processes tweets through all 5 reasoning tasks (R1-R5)
- Saves annotated CSV with reasoning traces

**Configuration:**
Edit `N_SAMPLES` in the script to control batch size.

**Usage:**
```bash
# Submit as SLURM job (recommended)
sbatch scripts/run_annotation.sh

# Or run directly (requires GPU)
python3 scripts/annotate.py
```

### run_annotation.sh
SLURM job script for batch annotation on GPU nodes.


**Monitor job:**
```bash
squeue -u $USER
tail -f annotation_conversational_*.log
```

## Utility Scripts

### preflight_check.py
**NEW:** Comprehensive pre-flight verification before starting annotation job.

Verifies all components are ready:
- Required files exist (data, scripts, model files)
- Python dependencies installed
- Qwen2.5-72B model cached locally
- Data structure valid
- Annotation configuration appropriate
- SLURM configuration correct

**Usage:**
```bash
# Run before submitting annotation job
python3 scripts/preflight_check.py
```

**Example output:**
```
✓ Files: PASSED
✓ Dependencies: PASSED
✓ Model Cache: PASSED (135.4 GB)
✓ Data Structure: PASSED (90,000 tweets)
✓ Annotation Config: PASSED (5,000 samples, ~7 days)
✓ SLURM Config: PASSED (168h, 40GB, 1 GPU)
```

### download_model.py
Pre-download Qwen2.5-72B model to local cache (run on login node with internet).

**Usage:**
```bash
python3 scripts/download_model.py
```

### view_annotations.py
View annotated reasoning traces from the command line.

**Usage:**
```bash
# View 3 examples (default)
python3 scripts/view_annotations.py

# View 10 examples
python3 scripts/view_annotations.py 10
```

### test_gpu.sh
Quick GPU availability test.

**Usage:**
```bash
bash scripts/test_gpu.sh
```

### start_jupyter_gpu.sh
Start Jupyter on a GPU node (for interactive development).

**Usage:**
```bash
bash scripts/start_jupyter_gpu.sh
```

## Data Preprocessing

### preprocess_syntactic.py
Parse tweets to CoNLL-U format using spaCy.

**Usage:**
```bash
# Process full dataset (~10-15 minutes)
cd data/COVIDSenti
python ../../scripts/preprocess_syntactic.py --input COVIDSenti.csv --output COVIDSenti_full_parsed.csv

# Process sample
python ../../scripts/preprocess_syntactic.py --input COVIDSenti.csv --output COVIDSenti_sample.csv --sample 1000
```

**Requirements:**
```bash
pip install spacy pandas tqdm
python -m spacy download en_core_web_sm
```

## End-to-End Workflow

### Step 1: Download Model (One-time setup)
Run on login node with internet access:
```bash
python3 scripts/download_model.py
```
Downloads Qwen2.5-72B-Instruct to `~/.cache/huggingface/hub/` (~135 GB)

### Step 2: Pre-Flight Check
**IMPORTANT:** Run this before submitting the annotation job to catch issues early:
```bash
python3 scripts/preflight_check.py
```
Verifies:
- All data files present (COVIDSenti.csv, COVIDSenti_full_parsed.csv)
- Dependencies installed (pandas, torch, transformers, bitsandbytes)
- Model cached locally
- Data structure valid (90,000 tweets with tweet, label, conllu_parse columns)
- Configuration appropriate (N_SAMPLES, time limit)

### Step 3: Submit Annotation Job
After all pre-flight checks pass:
```bash
sbatch scripts/run_annotation.sh
```

### Step 4: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Watch live log output
tail -f annotation_conversational_*.log

# The log will show:
# - Filtering progress (news vs conversational)
# - Model loading
# - Annotation progress with tqdm bar
# - Estimated completion time
```

### Step 5: View Results
When job completes (check with `squeue -u $USER`):
```bash
# View annotated reasoning traces
python3 scripts/view_annotations.py 10

# Or load in Jupyter
jupyter notebook 03_Teacher_Annotation.ipynb
```

Output file: `data/COVIDSenti/COVIDSenti_conversational_annotated_{N_SAMPLES}.csv`
