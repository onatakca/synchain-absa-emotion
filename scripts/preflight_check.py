#!/usr/bin/env python3
"""
Pre-flight check script for Syn-Chain ABSA annotation pipeline.
Run this BEFORE submitting the annotation job to verify everything is ready.

Usage:
    python3 scripts/preflight_check.py
"""

import sys
import os
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def print_error(text):
    print(f"{Colors.RED}✗{Colors.END} {text}")

def check_files():
    """Check if all required files exist."""
    print_header("1. CHECKING REQUIRED FILES")

    required_files = {
        "Data": [
            "data/COVIDSenti/COVIDSenti.csv",
            "data/COVIDSenti/COVIDSenti_full_parsed.csv",
        ],
        "Annotation": [
            "annotation/qwen_model.py",
            "annotation/prompts.py",
        ],
        "Scripts": [
            "scripts/annotate.py",
            "scripts/run_annotation.sh",
        ],
    }

    all_exist = True
    for category, files in required_files.items():
        print(f"\n{category} files:")
        for file_path in files:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                size_mb = size / (1024 * 1024)
                print_success(f"{file_path} ({size_mb:.1f} MB)")
            else:
                print_error(f"{file_path} NOT FOUND")
                all_exist = False

    return all_exist

def check_imports():
    """Check if all required Python packages are installed."""
    print_header("2. CHECKING PYTHON DEPENDENCIES")

    required_packages = [
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("tqdm", "tqdm"),
        ("bitsandbytes", "bitsandbytes"),
    ]

    all_installed = True
    for package, import_name in required_packages:
        try:
            __import__(import_name)
            print_success(f"{package}")
        except ImportError:
            print_error(f"{package} NOT INSTALLED")
            all_installed = False

    return all_installed

def check_model_cache():
    """Check if Qwen model is cached locally."""
    print_header("3. CHECKING MODEL CACHE")

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_pattern = "models--Qwen--Qwen2.5-72B-Instruct"

    if cache_dir.exists():
        model_dirs = list(cache_dir.glob(model_cache_pattern))
        if model_dirs:
            model_dir = model_dirs[0]
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.glob("*"))
                if snapshots:
                    snapshot_path = snapshots[0]
                    size = sum(f.stat().st_size for f in snapshot_path.rglob('*') if f.is_file())
                    size_gb = size / (1024 ** 3)
                    print_success(f"Qwen2.5-72B model found")
                    print(f"  Location: {snapshot_path}")
                    print(f"  Size: {size_gb:.1f} GB")
                    return True

    print_error("Qwen2.5-72B model NOT found in cache")
    print(f"  Expected location: {cache_dir}/{model_cache_pattern}")
    print(f"\n  Run this to download:")
    print(f"  {Colors.YELLOW}python3 scripts/download_model.py{Colors.END}")
    return False

def check_data_structure():
    """Check if parsed data has correct structure."""
    print_header("4. CHECKING DATA STRUCTURE")

    try:
        import pandas as pd

        df = pd.read_csv("data/COVIDSenti/COVIDSenti_full_parsed.csv")

        required_columns = ['tweet', 'label', 'conllu_parse']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print_error(f"Missing columns: {missing_columns}")
            return False

        print_success(f"All required columns present: {required_columns}")
        print(f"  Total rows: {len(df):,}")

        # Check for missing values
        missing_counts = df[required_columns].isnull().sum()
        if missing_counts.any():
            print_warning(f"Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"    {col}: {count} missing")
        else:
            print_success("No missing values")

        # Check label distribution
        label_counts = df['label'].value_counts()
        print(f"\n  Label distribution:")
        for label, count in label_counts.items():
            print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")

        return True

    except Exception as e:
        print_error(f"Error reading data: {e}")
        return False

def check_annotation_config():
    """Check annotation script configuration."""
    print_header("5. CHECKING ANNOTATION CONFIGURATION")

    try:
        with open("scripts/annotate.py", "r") as f:
            content = f.read()

        # Extract N_SAMPLES
        for line in content.split('\n'):
            if line.strip().startswith('N_SAMPLES ='):
                n_samples_line = line
                n_samples = int(line.split('=')[1].strip().split('#')[0])
                print_success(f"N_SAMPLES = {n_samples:,}")

                # Estimate time
                time_per_tweet = 2  # minutes
                total_minutes = n_samples * time_per_tweet
                total_hours = total_minutes / 60
                total_days = total_hours / 24

                print(f"\n  Estimated time:")
                print(f"    @ 2 min/tweet: ~{total_hours:.1f} hours (~{total_days:.1f} days)")

                if total_hours > 168:
                    print_warning(f"This exceeds SLURM time limit (168 hours)")
                    print(f"    Consider reducing N_SAMPLES or increasing time limit")

                break

        # Check GPU requirement
        if "RuntimeError: No GPU detected!" in content:
            print_success("GPU check enabled (will fail on login node - expected)")

        return True

    except Exception as e:
        print_error(f"Error reading annotation config: {e}")
        return False

def check_slurm_config():
    """Check SLURM job configuration."""
    print_header("6. CHECKING SLURM CONFIGURATION")

    try:
        with open("scripts/run_annotation.sh", "r") as f:
            content = f.read()

        config = {}
        for line in content.split('\n'):
            if line.startswith('#SBATCH'):
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[1].split('=')[0]
                    value = '='.join(parts[1].split('=')[1:]) if '=' in parts[1] else parts[2] if len(parts) > 2 else ''
                    config[key] = value

        print_success("SLURM configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Check if time limit is appropriate
        if '--time' in config:
            time_str = config['--time']
            hours = int(time_str.split(':')[0])
            if hours >= 168:
                print_success(f"Time limit: {hours} hours (sufficient for large batches)")
            else:
                print_warning(f"Time limit: {hours} hours (may be short for large batches)")

        return True

    except Exception as e:
        print_error(f"Error reading SLURM config: {e}")
        return False

def main():
    """Run all pre-flight checks."""
    print(f"\n{Colors.BOLD}PRE-FLIGHT CHECK FOR SYN-CHAIN ABSA ANNOTATION{Colors.END}")
    print(f"{Colors.BOLD}Run this before: sbatch scripts/run_annotation.sh{Colors.END}")

    checks = [
        ("Files", check_files),
        ("Dependencies", check_imports),
        ("Model Cache", check_model_cache),
        ("Data Structure", check_data_structure),
        ("Annotation Config", check_annotation_config),
        ("SLURM Config", check_slurm_config),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"Check failed with exception: {e}")
            results[name] = False

    # Print summary
    print_header("SUMMARY")

    all_passed = True
    for name, passed in results.items():
        if passed:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
            all_passed = False

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED!{Colors.END}")
        print(f"\n{Colors.GREEN}Ready to start annotation:{Colors.END}")
        print(f"  {Colors.BOLD}sbatch scripts/run_annotation.sh{Colors.END}")
        print(f"\nMonitor with:")
        print(f"  squeue -u $USER")
        print(f"  tail -f annotation_conversational_*.log")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME CHECKS FAILED{Colors.END}")
        print(f"\n{Colors.RED}Please fix the issues above before starting annotation.{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
