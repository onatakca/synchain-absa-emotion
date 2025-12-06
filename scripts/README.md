# Preprocessing Scripts

## preprocess_syntactic.py

Parses tweets to CoNLL-U format using spaCy.

**Usage:**
```bash
# Process full dataset (takes ~10-15 minutes)
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
