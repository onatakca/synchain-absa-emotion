# Syn-Chain ABSA with Emotion Detection

Extended implementation of **Syntax-Opinion-Sentiment Reasoning Chain (Syn-Chain)** for Aspect-Based Sentiment Analysis (ABSA) with emotion classification on COVID-19 tweets.

## Overview

This project extends the original Syn-Chain methodology with:
- **Aspect extraction** (R2): Identify COVID-19 related aspects (e.g., vaccines, masks, lockdown)
- **Emotion classification** (R5): Classify emotions beyond sentiment (fear, anger, joy, sadness, etc.)
- **Conversational focus**: Filter dataset to analyze opinions/emotions (not news headlines)
- **Knowledge distillation**: Teacher (Qwen2.5-72B) → Student (LLaMA-3-8B)

### Five-Task Reasoning Chain

1. **R1 - Syntactic Parsing**: Analyze grammatical structure (CoNLL-U format)
2. **R2 - Aspect Extraction**: Identify COVID-19 related aspects/targets
3. **R3 - Opinion Extraction**: Extract opinion expressions and their relationships
4. **R4 - Sentiment Classification**: Classify sentiment polarity (positive/negative/neutral)
5. **R5 - Emotion Classification**: Classify fine-grained emotions (fear, anger, joy, etc.)

## Repository Structure

```
synchain-absa-emotion/
├── 00_Data_Download.ipynb          # Download COVIDSenti dataset
├── 01_Data_Exploration.ipynb       # Explore data + conversational filtering
├── 02_Syntactic_Parsing.ipynb      # Parse tweets with Stanza
├── 03_Teacher_Annotation.ipynb     # Analyze annotated results
│
├── annotation/                     # Teacher model components
│   ├── qwen_model.py              # Qwen2.5-72B loader (4-bit quantized)
│   └── prompts.py                 # Task prompts (R1-R5)
│
├── scripts/                        # Batch execution scripts
│   ├── annotate.py                # Generate reasoning traces (conversational only)
│   ├── run_annotation.sh          # SLURM job script
│   ├── download_model.py          # Pre-download Qwen model
│   ├── view_annotations.py        # CLI viewer for results
│   └── README.md                  # Script documentation
│
├── modeling/                       # Student model (future training)
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
└── data/                           # Datasets
    └── COVIDSenti/
        ├── COVIDSenti_full_parsed.csv
        └── COVIDSenti_conversational_annotated_*.csv
```

## Quick Start

### 1. Download Model (one-time, on login node)
```bash
python3 scripts/download_model.py
```

### 2. Submit Annotation Job
```bash
sbatch scripts/run_annotation.sh
```

### 3. Monitor Progress
```bash
squeue -u $USER
tail -f annotation_conversational_*.log
```

### 4. View Results
```bash
# Command line viewer
python3 scripts/view_annotations.py 10

# Or use Jupyter notebook
jupyter notebook 03_Teacher_Annotation.ipynb
```

## Dataset

**COVIDSenti**: 90,000 COVID-19 tweets (Naseem et al., 2021)
- **Conversational tweets**: 38,277 (42.5%) - opinions, questions, emotions
- **News-like tweets**: 51,723 (57.5%) - headlines, links, factual statements

This project focuses on **conversational tweets** for meaningful ABSA and emotion analysis.
