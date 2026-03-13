#!/usr/bin/env python3
"""
Validate setup for HPC Multi-Agent Scheduler.
Checks dependencies, data files, and Ollama connectivity.
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, "data")


def check_dependencies():
    print("Checking Python dependencies...")
    required = {"numpy": "numpy", "pandas": "pandas", "requests": "requests"}
    missing = []
    for name, pkg in required.items():
        try:
            __import__(pkg)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}  (missing)")
            missing.append(name)
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def check_data_files():
    print("\nChecking data files...")
    required = {
        "Applications.csv": "App specs (CPU, IO, NW, Memory, Instances)",
        "Machines.csv":     "Machine specs (CPU, IO, NW, Memory, P_idle, P_max)",
        "Affinity.csv":     "Affinity matrix  (apps × machines, values 0/1)",
        "AntiAffinity.csv": "Anti-affinity matrix (apps × machines, values 0/1)",
    }
    missing = []
    for fname, desc in required.items():
        fpath = os.path.join(_DATA_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  ✓ data/{fname:<22} ({size:,} bytes)  — {desc}")
        else:
            print(f"  ✗ data/{fname:<22}  MISSING  — {desc}")
            missing.append(fname)
    if missing:
        print(f"\n  Place the missing CSV files in:  {_DATA_DIR}/")
        print("  Or run with --synthetic to use randomly generated data.")
        return False
    return True


def validate_data_format():
    print("\nValidating data format...")
    try:
        import pandas as pd
        import numpy as np

        apps     = pd.read_csv(os.path.join(_DATA_DIR, "Applications.csv"))
        machines = pd.read_csv(os.path.join(_DATA_DIR, "Machines.csv"))
        affinity = pd.read_csv(os.path.join(_DATA_DIR, "Affinity.csv"), index_col=0)
        forbidden= pd.read_csv(os.path.join(_DATA_DIR, "AntiAffinity.csv"), index_col=0)

        for name, df, cols in [
            ("Applications", apps,     ["CPU","IO","NW","Memory","Instances"]),
            ("Machines",     machines, ["CPU","IO","NW","Memory","P_idle","P_max"]),
        ]:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"  ✗ {name}.csv missing columns: {missing}")
                return False
            print(f"  ✓ {name}.csv  — {len(df)} rows, columns OK")

        n, m = len(apps), len(machines)
        for name, mat in [("Affinity", affinity), ("AntiAffinity", forbidden)]:
            if mat.shape != (n, m):
                print(f"  ✗ {name}.csv shape {mat.shape} ≠ expected ({n}, {m})")
                return False
            vals = set(mat.values.flatten())
            if not vals.issubset({0, 1}):
                print(f"  ⚠ {name}.csv contains values other than 0/1: {vals - {0,1}}")
            print(f"  ✓ {name}.csv  — shape ({n}, {m}), values OK")

        total_instances = int(apps["Instances"].sum())
        print(f"\n  Summary: {n} apps | {total_instances} total instances | {m} machines")
        return True

    except Exception as e:
        print(f"  ✗ Validation error: {e}")
        return False


def check_ollama(model: str = "llama3"):
    print(f"\nChecking Ollama connection (model: {model})...")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            print(f"  ✗ Ollama returned HTTP {resp.status_code}")
            print("    Is Ollama running?  Try: ollama serve")
            return False

        data = resp.json()
        available = [m["name"] for m in data.get("models", [])]
        if not available:
            print("  ✗ No models pulled yet.")
            print(f"    Run: ollama pull {model}")
            return False

        # Check if requested model is available (fuzzy match)
        found = any(model.split(":")[0] in a for a in available)
        print(f"  ✓ Ollama running. Available models: {', '.join(available)}")
        if found:
            print(f"  ✓ Model '{model}' is available")
        else:
            print(f"  ⚠ Model '{model}' not found. Available: {available}")
            print(f"    Run: ollama pull {model}")
        return True

    except Exception as e:
        print(f"  ✗ Could not reach Ollama: {e}")
        print("    Make sure Ollama is running:  ollama serve")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate HPC Scheduler setup")
    parser.add_argument("--model", default="llama3",
                        help="Ollama model to check (default: llama3)")
    parser.add_argument("--skip-ollama", action="store_true",
                        help="Skip Ollama connectivity check")
    args = parser.parse_args()

    print("=" * 70)
    print("HPC Multi-Agent Scheduler — Setup Validation")
    print("=" * 70)
    print()

    results = {}
    results["Dependencies"] = check_dependencies()
    results["Data files"]   = check_data_files()
    if results["Data files"]:
        results["Data format"] = validate_data_format()
    if not args.skip_ollama:
        results["Ollama"] = check_ollama(args.model)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("✓ All checks passed — ready to run!")
        print()
        print("Quick start:")
        print(f"  python demo_scheduler.py --model {args.model}")
        print(f"  python run_full_schedule.py --model {args.model}")
        print(f"  python run_ablation.py --model {args.model} --synthetic")
    else:
        print("✗ Fix the issues above, then re-run validate_setup.py")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
