# HPC Multi-Agent LLM Scheduler

A research framework for **power-aware VM/container placement** on HPC clusters using local LLMs via [Ollama](https://ollama.com). No API keys required — runs entirely on your machine.

The scheduler poses the placement problem to one or more LLM agents, which read structured candidate tables and output JSON placement decisions. An optional **Rearrangement Agent (RA)** performs post-placement Pareto-improving swaps. A **Debate** variant pits a Power Advocate against an Affinity Advocate before a Placement Judge makes the final call.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Variant A: PLACER + RA  (hpc_scheduler_agents.py)          │
│                                                             │
│  PLACER ──► placement decision (Rule T)                     │
│  RA     ──► optional Pareto-improving Move / Swap           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Variant B: Debate  (hpc_debate_agents.py)                  │
│                                                             │
│  POWER_ADVOCATE ──┐                                         │
│                   ├──► N debate rounds ──► PLACEMENT_JUDGE  │
│  AFFINITY_ADVOCATE┘                            │            │
│                                               RA            │
└─────────────────────────────────────────────────────────────┘

Ablation variants (no RA, power-only, affinity-only):
  hpc_no_ra.py  |  hpc_power_only.py  |  hpc_affinity_only.py
```

### Placement Policy — Rule T

At each step the scheduler builds a **candidate table** (all feasible placements sorted by incremental power ΔP):

```
epsilon = max(eps_rel × |ΔP_min|,  eps_min)

if any candidate has affinity=1 AND ΔP ≤ ΔP_min + epsilon:
    pick the affinity=1 candidate with lowest ΔP
else:
    pick the candidate with lowest ΔP
```

### Power Model

```
d = min(1,  0.70·u_cpu³ + 0.10·u_io + 0.05·u_nw + 0.15·u_mem)
P(m) = P_idle + d·(P_max − P_idle)
ΔP   = P(m)_after − P(m)_before
```

---

## Prerequisites

### 1. Install Ollama

Download from [ollama.com/download](https://ollama.com/download) or:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a model

```bash
ollama pull llama3          # 8B — fast, good baseline
ollama pull llama3.1        # 8B — better instruction following
ollama pull deepseek-r1     # 7B — strong reasoning
ollama pull deepseek-coder:6.7b  # good at structured JSON
ollama pull qwen3-next      # 235B MoE — best quality, needs ~50 GB VRAM
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. No GPU needed on the Python side — inference runs inside Ollama.

---

## Quick Start

```bash
# Validate everything is set up correctly
python validate_setup.py --model llama3

# Quick demo with synthetic data (8 apps, 5 machines)
python demo_scheduler.py --model llama3

# Full run on the bundled dataset
python run_full_schedule.py --model llama3

# Ablation study (no debate / 3-round / 5-round debate)
python run_ablation.py --model llama3 --synthetic --num-apps 10 --num-machines 20
```

### Example: validate_setup.py

```
======================================================================
HPC Multi-Agent Scheduler — Setup Validation
======================================================================

Checking Python dependencies...
  ✓ numpy
  ✓ pandas
  ✓ requests

Checking data files...
  ✓ data/Applications.csv     (12,847 bytes)  — App resource requirements
  ✓ data/Machines.csv         ( 4,201 bytes)  — Machine capacities + power profile
  ✓ data/Affinity.csv         (38,920 bytes)  — Binary affinity matrix
  ✓ data/AntiAffinity.csv     (38,920 bytes)  — Binary anti-affinity matrix

Validating data format...
  ✓ Applications.csv  — 50 rows, columns OK
  ✓ Machines.csv      — 30 rows, columns OK
  ✓ Affinity.csv      — shape (50, 30), values OK
  ✓ AntiAffinity.csv  — shape (50, 30), values OK

  Summary: 50 apps | 127 total instances | 30 machines

Checking Ollama connection (model: llama3)...
  ✓ Ollama running. Available models: llama3:latest, llama3.1:latest, deepseek-r1:latest
  ✓ Model 'llama3' is available

======================================================================
SUMMARY
======================================================================
  ✓ PASS  Dependencies
  ✓ PASS  Data files
  ✓ PASS  Data format
  ✓ PASS  Ollama

✓ All checks passed — ready to run!

Quick start:
  python demo_scheduler.py --model llama3
  python run_full_schedule.py --model llama3
  python run_ablation.py --model llama3 --synthetic
```

### Example: demo_scheduler.py

```
================================================================================
HPC Multi-Agent Scheduler — Demo
================================================================================

Configuration:
  Apps:       8
  Machines:   5
  Seed:       42
  Model:      llama3
  Lambda:     0.5
  Max iter:   30

Generated: 8 apps, 5 machines, 18 total instances

[t=0] ======================================================================
[t=0] HPC Scheduler — PLACER + RA (Power primary, Affinity secondary)
[t=0] ======================================================================
[t=0] Calling PLACER...
[t=0] Placed inst_0: app_3→m2 (aff=1 ΔP=0.0312W)
[t=0] Calling RA...
[t=0] RA: no Pareto-improving ops — skip LLM.
[t=1] Calling PLACER...
[t=1] Placed inst_1: app_0→m2 (aff=0 ΔP=0.0874W)
[t=1] Calling RA...
[t=1] RA Move: inst_0 m2→m4
[t=2] Calling PLACER...
[t=2] Placed inst_2: app_1→m0 (aff=1 ΔP=0.1205W)
[t=2] Calling RA...
[t=2] RA: no Pareto-improving ops — skip LLM.
...
[t=17] Placed inst_17: app_6→m3 (aff=0 ΔP=2.3841W)
[t=17] RA: no Pareto-improving ops — skip LLM.
[t=17] No eligible apps — stopping.

================================================================================
Final Results Summary
================================================================================
Placements made:     18/18
Total cluster power: 142.37 W
Affinity score:      11
Avg CPU utilization: 61.4%
Time elapsed:        38.52 s

Per-Machine Statistics:
--------------------------------------------------------------------------------
 Machine   CPU Util    Power (W)  Instances
       0      72.3%        31.84          4
       1      45.1%        24.17          3
       2      83.6%        38.92          5
       3      51.8%        26.40          3
       4      38.2%        21.04          3
```

### Example: run_full_schedule.py

```
================================================================================
HPC Multi-Agent Scheduler — Full Dataset
================================================================================
  Model      : llama3
  Ollama URL : http://localhost:11434
  Output dir : results/llama3/

Loading CSV files...
  Apps:          data/Applications.csv
  Machines:      data/Machines.csv
  Affinity:      data/Affinity.csv
  Anti-Affinity: data/AntiAffinity.csv

Dataset: 50 apps | 127 instances | 30 machines
Params : lambda=0.5  max_iter=100  eps_rel=0.0  eps_min=0.25

[t=0] ======================================================================
[t=0] HPC Scheduler — PLACER + RA (Power primary, Affinity secondary)
[t=0] ======================================================================
[t=0] Calling PLACER...
[t=0] Placed inst_0: app_12→m7 (aff=1 ΔP=0.0000W)
[t=0] Calling RA...
[t=0] RA: no Pareto-improving ops — skip LLM.
[t=1] Calling PLACER...
[t=1] Placed inst_1: app_12→m7 (aff=1 ΔP=0.0431W)
[t=1] Calling RA...
[t=1] RA: no Pareto-improving ops — skip LLM.
[t=2] Calling PLACER...
[t=2] Placed inst_2: app_5→m3  (aff=0 ΔP=0.0000W)
[t=2] Calling RA...
[t=2] RA Move: inst_1 m7→m14
...
[t=126] Placed inst_126: app_43→m22 (aff=1 ΔP=4.1073W)
[t=126] RA: no Pareto-improving ops — skip LLM.
[t=126] No eligible apps — stopping.
[t=126] ======================================================================
[t=126] Done. Placements=127  Swaps=14
[t=126] Power=1847.33W  Affinity=89  LLM calls=240  Time=312.4s
[t=126] CSVs → results/llama3/
[t=126] ======================================================================

================================================================================
FINAL RESULTS
================================================================================
  Placements   :    127 / 127  (100.0%)
  Cluster power: 1847.33 W
  Affinity pts :     89
  LLM calls    :    240  (312.4s)
  CPU util     : min=12.3%  mean=54.7%  max=91.2%
  Power per m  : min=62.1W  mean=61.6W  total=1847.3W
  Affinity rate: 70.1%  (89/127 placements)

Results saved to: results/llama3/
  decisions.csv  placements.csv  step_stats.csv  llm_stats.csv
```

### Example: run_ablation.py

```
Synthetic: 10 apps × 20 machines (seed=42)

Model      : llama3
Output dir : results/ablation_llama3
Ablation configs (3 total):
  no_debate                       debate=0  choices=1  eps_rel=0.0  eps_min=0.25
  debate_3r_c1                    debate=3  choices=1  eps_rel=0.0  eps_min=0.25
  debate_5r_c1                    debate=5  choices=1  eps_rel=0.0  eps_min=0.25

================================================================================
CONFIG: no_debate  (debate_rounds=0, num_choices=1, eps_rel=0.0, eps_min=0.25)
================================================================================
[t=0] HPC Debate Scheduler — debate_rounds=0  num_choices=1
[t=0] === Debate phase (rounds=0) ===
[t=0] Calling PLACEMENT JUDGE...
[t=0] Placed inst_0: app_2→m5 (aff=1 dP=0.0000W)
...
[t=0] Done. debate_rounds=0  Placements=22  Swaps=3

================================================================================
CONFIG: debate_3r_c1  (debate_rounds=3, num_choices=1, eps_rel=0.0, eps_min=0.25)
================================================================================
[t=0] === Debate phase (rounds=3) ===
[t=0]   Debate R1: Power Advocate (1 choices)...
[t=0]   Debate R1: Affinity Advocate (1 choices)...
[t=0]   Debate R2: Power Advocate (1 choices)...
[t=0]   Debate R2: Affinity Advocate (1 choices)...
[t=0]   Debate R3: Power Advocate (1 choices)...
[t=0]   Debate R3: Affinity Advocate (1 choices)...
[t=0] Calling PLACEMENT JUDGE...
[t=0] Placed inst_0: app_2→m5 (aff=1 dP=0.0000W)
...
[t=0] Done. debate_rounds=3  Placements=22  Swaps=2

...

========================================================================================================================
ABLATION RESULTS
========================================================================================================================

Config                         N    K     ε  Placed     Power   Aff%   LLM   Tokens  Time(s)
------------------------------------------------------------------------------------------------------------------------
no_debate                      0    1   0.0      22   4821.34  72.73     44    18240    91.34
debate_3r_c1                   3    1   0.0      22   4798.11  77.27    176    81630   387.21
debate_5r_c1                   5    1   0.0      22   4791.02  81.82    284   131820   623.47

Summary CSV    : results/ablation_llama3/ablation_summary.csv
Per-config CSVs: results/ablation_llama3/<config_name>/
  Each dir contains: decisions.csv, placements.csv,
  step_stats.csv, llm_stats.csv, conversations.csv
```

---

## Dataset

Place CSV files in `data/`. The repo ships with a sample dataset:

| File | Contents |
|------|----------|
| `data/Applications.csv` | App resource requirements: CPU, IO, NW, Memory, Instances |
| `data/Machines.csv` | Machine capacities + power profile: CPU, IO, NW, Memory, P_idle, P_max |
| `data/Affinity.csv` | Binary affinity matrix (apps × machines) |
| `data/AntiAffinity.csv` | Binary anti-affinity / forbidden matrix (apps × machines) |

All schedulers also support **synthetic data** via `--synthetic` (no CSV files needed).

---

## Scripts

### `run_full_schedule.py` — Main scheduler

Runs the PLACER + RA scheduler. Results go to `results/<model_name>/`.

```bash
# CSV data (default: data/ folder)
python run_full_schedule.py --model llama3.1

# Different model
python run_full_schedule.py --model deepseek-r1

# Synthetic data
python run_full_schedule.py --model llama3 --synthetic --num-apps 50 --num-machines 80

# Tune epsilon (affinity slack)
python run_full_schedule.py --model llama3 --eps-rel 0.1 --eps-min 0.5

# Custom output
python run_full_schedule.py --model llama3 --output my_results/

# Via env var
OLLAMA_MODEL=llama3.1 python run_full_schedule.py
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `llama3` | Ollama model name |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--num-apps` | all | Slice first N apps from CSV |
| `--num-machines` | all | Slice first M machines from CSV |
| `--synthetic` | off | Generate random data |
| `--max-iter` | 100 | Max placement iterations |
| `--eps-rel` | 0.0 | Relative affinity slack (fraction of ΔP_min) |
| `--eps-min` | 0.25 | Minimum affinity slack (Watts) |
| `--output` | `results/<model>/` | Output directory |

---

### `run_ablation.py` — Ablation study

Runs the Debate scheduler across multiple configurations. Results go to `results/ablation_<model_name>/`.

```bash
# Default: 3 configs (no_debate / debate_3r / debate_5r)
python run_ablation.py --model llama3

# Synthetic data
python run_ablation.py --model llama3 --synthetic --num-apps 10 --num-machines 20

# Custom debate rounds
python run_ablation.py --model llama3 --debate-configs 0 1 3 5 7

# Sweep epsilon
python run_ablation.py --model llama3 --eps-rel 0.0 0.1 0.2 0.5

# Single config
python run_ablation.py --model llama3 --debate-rounds 3 --num-choices 1
```

---

### `demo_scheduler.py` — Quick test

Runs on small synthetic data (8 apps, 5 machines by default):

```bash
python demo_scheduler.py --model llama3 --num-apps 20 --num-machines 10
python demo_scheduler.py --model deepseek-r1 --max-iter 20
```

---

### `analyze_results.py` — Result analysis

Detailed breakdown of any run's output JSON:

```bash
python analyze_results.py results/llama3/scheduler_results.json
```

---

### `validate_setup.py` — Setup check

```bash
python validate_setup.py --model llama3
python validate_setup.py --skip-ollama   # skip connectivity check
```

---

## Output Files

Every run writes to its output directory:

| File | Contents |
|------|----------|
| `decisions.csv` | Every LLM decision — action, app/machine IDs, power delta, affinity, rationale, tokens, timing |
| `placements.csv` | Final placement map: instance_id → app_id → machine_id |
| `step_stats.csv` | Per-timestep: placed count, cluster power, affinity score, LLM call count |
| `llm_stats.csv` | Aggregate LLM stats: total calls, tokens, time per agent |
| `conversations.csv` | *(Debate variant only)* Full prompt + response for every LLM turn |

Output directory layout by model:

```
results/
  llama3/
    decisions.csv
    placements.csv
    step_stats.csv
    llm_stats.csv
  llama3.1/
    decisions.csv
    ...
  ablation_llama3/
    no_debate/
      decisions.csv  placements.csv  step_stats.csv  llm_stats.csv  conversations.csv
    debate_3r_c1/
      ...
    debate_5r_c1/
      ...
    ablation_summary.csv
```

---

## Comparing Models

Run the same workload across multiple models and compare:

```bash
for model in llama3 llama3.1 deepseek-r1 deepseek-coder:6.7b; do
  echo "=== $model ==="
  python run_full_schedule.py --model $model --synthetic --num-apps 20 --num-machines 30 --quiet
done
```

Results land in `results/llama3/`, `results/llama3.1/`, `results/deepseek-r1/`, etc.

To compare `llm_stats.csv` and `step_stats.csv` across runs:

```python
import pandas as pd, pathlib

models = ["llama3", "llama3.1", "deepseek-r1"]
rows = []
for m in models:
    path = pathlib.Path(f"results/{m}/llm_stats.csv")
    if path.exists():
        df = pd.read_csv(path)
        df["model"] = m
        rows.append(df)

print(pd.concat(rows).to_string())
```

---

## Scheduler Variants

| File | Variant | Agents | RA |
|------|---------|--------|----|
| `hpc_scheduler_agents.py` | PLACER + RA *(default)* | 2 | ✓ |
| `hpc_debate_agents.py` | Debate + RA | 4 | ✓ |
| `hpc_no_ra.py` | PLACER only | 1 | ✗ |
| `hpc_power_only.py` | Power-only (no affinity) | 1 | ✗ |
| `hpc_affinity_only.py` | Affinity-only (no power) | 1 | ✗ |

Each file is standalone and can be run directly:

```bash
python hpc_scheduler_agents.py --model llama3 --synthetic --seed 42
python hpc_debate_agents.py    --model llama3 --debate-rounds 3 --synthetic
python hpc_power_only.py       --model llama3 --synthetic
```

---

## Model Recommendations

| Use case | Recommended model |
|----------|-------------------|
| Quick testing | `llama3` or `deepseek-coder:6.7b` |
| Best JSON reliability | `llama3.1` |
| Best reasoning quality | `deepseek-r1` or `qwen3-next` |
| Fastest throughput | `deepseek-coder:6.7b` |
| Large-scale research | `qwen3-next` (needs ~50 GB VRAM) |

If the LLM returns unparseable JSON, the scheduler falls back to a deterministic minimum-ΔP placement and logs it as `"deterministic fallback"` in `decisions.csv`. Larger / instruction-tuned models produce fewer fallbacks.

---

## Troubleshooting

**`Connection refused` on Ollama**
```bash
ollama serve          # start the server
# or on macOS: open the Ollama desktop app
```

**Model not found**
```bash
ollama list           # see what's pulled
ollama pull llama3    # pull the model
```

**Slow inference**
- Use `--max-iter 10` for quick tests
- `deepseek-coder:6.7b` is the fastest model for structured JSON tasks

**High fallback rate (many `deterministic fallback` entries)**
- Try a larger or more instruction-following model (`llama3.1`, `deepseek-r1`)
- Check that Ollama has enough memory — OOM causes truncated responses

---

## License

MIT
