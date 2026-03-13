#!/usr/bin/env python3
"""
Ablation Study Runner
======================
Runs the HPC Debate Scheduler with different debate-round configurations
and stores all results under  results/ablation_<model_name>/  by default,
so ablations across different LLMs stay cleanly separated.

Configurations (default):
  no_debate   : debate_rounds=0  (placement judge decides alone)
  debate_3r   : debate_rounds=3  (3 rounds Power vs Affinity debate)
  debate_5r   : debate_rounds=5  (5 rounds Power vs Affinity debate)

Output layout:
  results/ablation_<model>/
    no_debate/        decisions.csv  placements.csv  step_stats.csv
                      llm_stats.csv  conversations.csv
    debate_3r/        (same files)
    debate_5r/        (same files)
    ablation_summary.csv   ← comparative summary across all configs

Usage:
  # With CSV data files in data/ (default)
  python run_ablation.py --model llama3.1

  # With synthetic data
  python run_ablation.py --model deepseek-r1 --synthetic --num-apps 10 --num-machines 20

  # Custom debate configs
  python run_ablation.py --model llama3 --debate-configs 0 1 3 5

  # Sweep epsilon
  python run_ablation.py --model llama3 --eps-rel 0.0 0.1 0.2 0.5
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpc_debate_agents import (
    AppSpec, MachineState, SchedulerState,
    HPCDebateOrchestrator,
    generate_random_state,
)

# Default data directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, "data")


def _safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(" ", "_")


def deep_copy_state(state: SchedulerState) -> SchedulerState:
    """Deep-copy a SchedulerState so each ablation run starts fresh."""
    return SchedulerState(
        machines=[MachineState(
            machine_id=m.machine_id,
            cpu_cap=m.cpu_cap, io_cap=m.io_cap, nw_cap=m.nw_cap, mem_cap=m.mem_cap,
            p_idle=m.p_idle, p_max=m.p_max,
            cpu_used=0.0, io_used=0.0, nw_used=0.0, mem_used=0.0,
        ) for m in state.machines],
        apps=[AppSpec(
            app_id=a.app_id, cpu=a.cpu, io=a.io, nw=a.nw, mem=a.mem,
            instances_total=a.instances_total, instances_placed=0,
        ) for a in state.apps],
        affinity=state.affinity.copy(),
        forbidden=state.forbidden.copy(),
        timestep=0, _next_iid=0,
    )


SUMMARY_COLS = [
    "config_name", "model", "debate_rounds", "num_choices", "eps_rel", "eps_min",
    "total_placements", "total_instances", "completion_rate_pct",
    "total_swaps", "total_power_W", "total_affinity", "affinity_rate_pct",
    "llm_calls_total", "llm_calls_debate", "llm_tokens_total",
    "llm_tokens_placer", "llm_tokens_debate", "llm_tokens_ra",
    "llm_time_total_s", "llm_time_debate_s", "wall_time_s", "output_dir",
]


def run_single_config(config_name, debate_rounds, num_choices, base_state,
                      model, ollama_url, max_iter, eps_rel, eps_min,
                      verbose, base_output_dir) -> dict:
    output_dir = base_output_dir / config_name
    nc_label = "full_rank" if num_choices <= 0 else str(num_choices)
    print("\n" + "=" * 80)
    print(f"CONFIG: {config_name}  "
          f"(debate_rounds={debate_rounds}, num_choices={nc_label}, "
          f"eps_rel={eps_rel}, eps_min={eps_min})")
    print("=" * 80)

    state = deep_copy_state(base_state)
    total_instances = sum(a.instances_total for a in state.apps)

    t0 = time.time()
    orch = HPCDebateOrchestrator(
        state=state, model=model, ollama_url=ollama_url,
        max_iterations=max_iter,
        eps_rel=eps_rel, eps_min=eps_min,
        debate_rounds=debate_rounds, num_choices=num_choices,
        verbose=verbose, output_dir=str(output_dir),
    )
    results = orch.run()
    wall_time = time.time() - t0

    aff_scores = [p.affinity for p in state.history]
    aff_rate   = 100.0 * sum(aff_scores) / len(aff_scores) if aff_scores else 0.0
    completion = 100.0 * results["total_placements"] / total_instances if total_instances else 0.0

    return {
        "config_name":        config_name,
        "model":              model,
        "debate_rounds":      debate_rounds,
        "num_choices":        num_choices,
        "eps_rel":            eps_rel,
        "eps_min":            eps_min,
        "total_placements":   results["total_placements"],
        "total_instances":    total_instances,
        "completion_rate_pct":f"{completion:.2f}",
        "total_swaps":        results["total_swaps"],
        "total_power_W":      f"{results['total_power_W']:.2f}",
        "total_affinity":     results["total_affinity"],
        "affinity_rate_pct":  f"{aff_rate:.2f}",
        "llm_calls_total":    results["llm_calls"],
        "llm_calls_debate":   results["llm_calls_debate"],
        "llm_tokens_total":   results["llm_tokens_total"],
        "llm_tokens_placer":  results["llm_tokens_placer"],
        "llm_tokens_debate":  results["llm_tokens_debate"],
        "llm_tokens_ra":      results["llm_tokens_ra"],
        "llm_time_total_s":   f"{results['llm_time_s']:.2f}",
        "llm_time_debate_s":  f"{results['llm_time_debate_s']:.2f}",
        "wall_time_s":        f"{wall_time:.2f}",
        "output_dir":         str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study — HPC Debate Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source
    parser.add_argument("--apps",          default=os.path.join(_DATA_DIR, "Applications.csv"))
    parser.add_argument("--machines",      default=os.path.join(_DATA_DIR, "Machines.csv"))
    parser.add_argument("--affinity",      default=os.path.join(_DATA_DIR, "Affinity.csv"))
    parser.add_argument("--forbidden",     default=os.path.join(_DATA_DIR, "AntiAffinity.csv"))
    parser.add_argument("--num-apps",      type=int, default=None)
    parser.add_argument("--num-machines",  type=int, default=None)
    parser.add_argument("--synthetic",     action="store_true")
    parser.add_argument("--seed",          type=int, default=42)

    # Scheduler
    parser.add_argument("--model",         default=os.environ.get("OLLAMA_MODEL", "llama3"))
    parser.add_argument("--ollama-url",    default="http://localhost:11434")
    parser.add_argument("--max-iter",      type=int,   default=1000)
    parser.add_argument("--eps-rel",       type=float, nargs="+", default=[0.0])
    parser.add_argument("--eps-min",       type=float, nargs="+", default=[0.25])
    parser.add_argument("--verbose",       action="store_true")

    # Ablation configs
    parser.add_argument("--debate-configs",  type=int, nargs="+", default=[0, 3, 5],
                        help="debate_rounds values to test (default: 0 3 5)")
    parser.add_argument("--debate-rounds",   type=int, default=None)
    parser.add_argument("--choices-configs", type=int, nargs="+", default=[1, 3, 0],
                        help="num_choices values (1=single, 3=top-3, 0=full rank)")
    parser.add_argument("--num-choices",     type=int, default=None)
    parser.add_argument("--output-dir",      default=None,
                        help="Base output dir (default: results/ablation_<model>/)")

    args = parser.parse_args()

    # Auto output dir
    base_output = Path(args.output_dir or os.path.join(
        _SCRIPT_DIR, "results", f"ablation_{_safe_model_name(args.model)}"
    ))
    base_output.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.synthetic:
        num_apps     = args.num_apps     or 10
        num_machines = args.num_machines or 20
        base_state = generate_random_state(num_apps, num_machines, seed=args.seed)
        print(f"Synthetic: {num_apps} apps × {num_machines} machines (seed={args.seed})")
    else:
        apps_df      = pd.read_csv(args.apps)
        machines_df  = pd.read_csv(args.machines)
        affinity_df  = pd.read_csv(args.affinity,  index_col=0)
        forbidden_df = pd.read_csv(args.forbidden, index_col=0)

        num_apps     = min(args.num_apps     or len(apps_df),     len(apps_df))
        num_machines = min(args.num_machines or len(machines_df), len(machines_df))

        apps_df     = apps_df.iloc[:num_apps]
        machines_df = machines_df.iloc[:num_machines]
        aff_matrix  = affinity_df.iloc[:num_apps, :num_machines].values.astype(int)
        forb_matrix = forbidden_df.iloc[:num_apps, :num_machines].values.astype(int)

        apps = [AppSpec(app_id=i, cpu=r["CPU"], io=r["IO"], nw=r["NW"],
                        mem=r["Memory"], instances_total=int(r["Instances"]))
                for i, r in enumerate(apps_df.to_dict("records"))]
        machines = [MachineState(machine_id=i, cpu_cap=r["CPU"], io_cap=r["IO"],
                                 nw_cap=r["NW"], mem_cap=r["Memory"],
                                 p_idle=r["P_idle"], p_max=r["P_max"])
                    for i, r in enumerate(machines_df.to_dict("records"))]
        base_state = SchedulerState(machines=machines, apps=apps,
                                    affinity=aff_matrix, forbidden=forb_matrix)
        total_inst = sum(a.instances_total for a in apps)
        print(f"CSV: {num_apps} apps × {num_machines} machines ({total_inst} instances)")

    # ── Resolve eps pairs ─────────────────────────────────────────────────────
    eps_rel_list = args.eps_rel
    eps_min_list = args.eps_min
    if len(eps_min_list) == 1 and len(eps_rel_list) > 1:
        eps_min_list = eps_min_list * len(eps_rel_list)
    elif len(eps_min_list) != len(eps_rel_list):
        parser.error("--eps-min must be 1 value or match --eps-rel count")
    eps_pairs = list(zip(eps_rel_list, eps_min_list))

    # ── Resolve debate / choice configs ──────────────────────────────────────
    debate_list  = [args.debate_rounds] if args.debate_rounds is not None else sorted(args.debate_configs)
    choices_list = [args.num_choices]   if args.num_choices  is not None else args.choices_configs

    # Cross-product: debate_rounds × num_choices × eps pairs
    configs = []
    for dr in debate_list:
        nc_list = [1] if dr == 0 else choices_list
        for nc in nc_list:
            for eps_r, eps_m in eps_pairs:
                nc_tag = "full" if nc <= 0 else str(nc)
                if len(eps_pairs) > 1:
                    name = f"dr{dr}_c{nc_tag}_epsrel{int(eps_r*100)}pct"
                elif dr == 0:
                    name = "no_debate"
                else:
                    name = f"debate_{dr}r_c{nc_tag}"
                configs.append((name, dr, nc, eps_r, eps_m))

    print(f"\nModel      : {args.model}")
    print(f"Output dir : {base_output}")
    print(f"Ablation configs ({len(configs)} total):")
    for name, dr, nc, er, em in configs:
        nc_label = "full_rank" if nc <= 0 else str(nc)
        print(f"  {name:<30} debate={dr}  choices={nc_label}  eps_rel={er}  eps_min={em}")
    print()

    # ── Run ablations ─────────────────────────────────────────────────────────
    summaries = []
    for config_name, debate_rounds, num_choices, eps_rel, eps_min in configs:
        row = run_single_config(
            config_name=config_name, debate_rounds=debate_rounds,
            num_choices=num_choices, base_state=base_state,
            model=args.model, ollama_url=args.ollama_url,
            max_iter=args.max_iter, eps_rel=eps_rel, eps_min=eps_min,
            verbose=args.verbose, base_output_dir=base_output,
        )
        summaries.append(row)

    # ── Write summary CSV ─────────────────────────────────────────────────────
    summary_path = base_output / "ablation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for row in summaries:
            w.writerow(row)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("ABLATION RESULTS")
    print("=" * 120)
    print(f"\n{'Config':<30} {'N':>3} {'K':>4} {'ε':>5} "
          f"{'Placed':>7} {'Power':>9} {'Aff%':>6} "
          f"{'LLM':>5} {'Tokens':>8} {'Time(s)':>8}")
    print("-" * 120)
    for s in summaries:
        nc_label = "all" if int(s["num_choices"]) <= 0 else str(s["num_choices"])
        print(f"{s['config_name']:<30} {s['debate_rounds']:>3} {nc_label:>4} "
              f"{s['eps_rel']:>5} "
              f"{s['total_placements']:>7} {s['total_power_W']:>9} "
              f"{s['affinity_rate_pct']:>6} "
              f"{s['llm_calls_total']:>5} {s['llm_tokens_total']:>8} "
              f"{s['wall_time_s']:>8}")

    print(f"\nSummary CSV    : {summary_path}")
    print(f"Per-config CSVs: {base_output}/<config_name>/")
    print("  Each dir contains: decisions.csv, placements.csv,")
    print("  step_stats.csv, llm_stats.csv, conversations.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
