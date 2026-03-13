#!/usr/bin/env python3
"""
Run HPC Multi-Agent Scheduler — Full Dataset or Synthetic

Results are stored under  results/<model_name>/  by default, so running
multiple models keeps their outputs cleanly separated.

Usage examples:
  # Synthetic (no CSV files needed)
  python run_full_schedule.py --model llama3.1 --num-apps 20 --num-machines 30

  # From CSV files in data/ (default paths)
  python run_full_schedule.py --model deepseek-r1

  # From custom CSV files
  python run_full_schedule.py --model qwen3-next:latest \\
      --apps /path/to/Applications.csv \\
      --machines /path/to/Machines.csv \\
      --affinity /path/to/Affinity.csv \\
      --forbidden /path/to/AntiAffinity.csv

  # Override output dir
  python run_full_schedule.py --model llama3 --output my_results/

  # Tune via env var
  OLLAMA_MODEL=llama3.1 python run_full_schedule.py
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from hpc_scheduler_agents import (
    load_data_from_csv,
    generate_random_state,
    HPCSchedulerOrchestrator,
)

# Default data directory (relative to this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, "data")


def _safe_model_name(model: str) -> str:
    """Convert model name to a filesystem-safe string for output dirs."""
    return model.replace(":", "_").replace("/", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        description="HPC Multi-Agent Scheduler — Full Run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source
    data = parser.add_argument_group("Data source")
    data.add_argument("--num-apps",    type=int, default=None,
                      help="Number of apps  (synthetic mode, or slice of CSV)")
    data.add_argument("--num-machines",type=int, default=None,
                      help="Number of machines (synthetic mode, or slice of CSV)")
    data.add_argument("--seed",        type=int, default=None,
                      help="RNG seed for synthetic data")
    data.add_argument("--synthetic",   action="store_true",
                      help="Generate random data instead of loading CSVs")
    data.add_argument("--apps",     default=os.path.join(_DATA_DIR, "Applications.csv"),
                      help="Applications CSV (default: data/Applications.csv)")
    data.add_argument("--machines", default=os.path.join(_DATA_DIR, "Machines.csv"),
                      help="Machines CSV     (default: data/Machines.csv)")
    data.add_argument("--affinity", default=os.path.join(_DATA_DIR, "Affinity.csv"),
                      help="Affinity CSV     (default: data/Affinity.csv)")
    data.add_argument("--forbidden",default=os.path.join(_DATA_DIR, "AntiAffinity.csv"),
                      help="Anti-Affinity CSV (default: data/AntiAffinity.csv)")

    # Scheduler
    parser.add_argument("--model",     default=os.environ.get("OLLAMA_MODEL", "llama3"),
                        help="Ollama model (default: llama3 or $OLLAMA_MODEL)")
    parser.add_argument("--ollama-url",default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--lambda",    dest="lambda_weight", type=float, default=0.5,
                        help="Lambda weight 0=affinity-only … 1=power-only (default: 0.5)")
    parser.add_argument("--max-iter",  type=int, default=100,
                        help="Max scheduling iterations (default: 100)")
    parser.add_argument("--eps-rel",   type=float, default=0.0)
    parser.add_argument("--eps-min",   type=float, default=0.25)
    parser.add_argument("--output",    default=None,
                        help="Output directory (default: results/<model_name>/)")
    parser.add_argument("--quiet",     action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Auto output dir: results/<model_name>/
    output_dir = args.output or os.path.join(
        _SCRIPT_DIR, "results", _safe_model_name(args.model)
    )

    print("=" * 80)
    print("HPC Multi-Agent Scheduler — Full Dataset")
    print("=" * 80)
    print(f"  Model      : {args.model}")
    print(f"  Ollama URL : {args.ollama_url}")
    print(f"  Output dir : {output_dir}")
    print()

    # ── Data loading ──────────────────────────────────────────────────────────
    if args.synthetic:
        num_apps     = args.num_apps     or 200
        num_machines = args.num_machines or 250
        print(f"Generating synthetic data: {num_apps} apps × {num_machines} machines "
              f"(seed={args.seed})")
        state = generate_random_state(num_apps, num_machines, seed=args.seed)

    else:
        for path, label in [(args.apps, "Apps"), (args.machines, "Machines"),
                            (args.affinity, "Affinity"), (args.forbidden, "AntiAffinity")]:
            if not os.path.exists(path):
                print(f"ERROR: {label} file not found: {path}")
                print("  → Place CSVs in data/ or pass --apps/--machines/--affinity/--forbidden")
                print("  → Or use --synthetic for random data")
                return 1

        print("Loading CSV files...")
        print(f"  Apps:        {args.apps}")
        print(f"  Machines:    {args.machines}")
        print(f"  Affinity:    {args.affinity}")
        print(f"  Anti-Affinity: {args.forbidden}")
        print()

        import pandas as pd
        apps_df      = pd.read_csv(args.apps)
        machines_df  = pd.read_csv(args.machines)
        affinity_df  = pd.read_csv(args.affinity,  index_col=0)
        forbidden_df = pd.read_csv(args.forbidden, index_col=0)

        num_apps     = min(args.num_apps     or len(apps_df),     len(apps_df))
        num_machines = min(args.num_machines or len(machines_df), len(machines_df))

        apps_df     = apps_df.iloc[:num_apps]
        machines_df = machines_df.iloc[:num_machines]

        from hpc_scheduler_agents import AppSpec, MachineState, SchedulerState
        apps = [AppSpec(app_id=i, cpu=r["CPU"], io=r["IO"], nw=r["NW"],
                        mem=r["Memory"], instances_total=int(r["Instances"]))
                for i, r in enumerate(apps_df.to_dict("records"))]
        machines = [MachineState(machine_id=i, cpu_cap=r["CPU"], io_cap=r["IO"],
                                 nw_cap=r["NW"], mem_cap=r["Memory"],
                                 p_idle=r["P_idle"], p_max=r["P_max"])
                    for i, r in enumerate(machines_df.to_dict("records"))]
        affinity  = affinity_df.iloc[:num_apps, :num_machines].values.astype(int)
        forbidden = forbidden_df.iloc[:num_apps, :num_machines].values.astype(int)
        state = SchedulerState(machines=machines, apps=apps,
                               affinity=affinity, forbidden=forbidden)

    total_instances = sum(app.instances_total for app in state.apps)
    total_apps      = len(state.apps)
    total_machines  = len(state.machines)

    print(f"Dataset: {total_apps} apps | {total_instances} instances | {total_machines} machines")
    print(f"Params : lambda={args.lambda_weight}  max_iter={args.max_iter}  "
          f"eps_rel={args.eps_rel}  eps_min={args.eps_min}")
    print()

    orchestrator = HPCSchedulerOrchestrator(
        state=state,
        model=args.model,
        ollama_url=args.ollama_url,
        lambda_weight=args.lambda_weight,
        max_iterations=args.max_iter,
        eps_rel=args.eps_rel,
        eps_min=args.eps_min,
        verbose=not args.quiet,
        output_dir=output_dir,
    )

    results = orchestrator.run()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    completion_rate = (100 * results["total_placements"] / total_instances
                       if total_instances else 0)

    print(f"  Placements   : {results['total_placements']:6d} / {total_instances}"
          f"  ({completion_rate:.1f}%)")
    print(f"  Cluster power: {results['total_power_W']:8.2f} W")
    print(f"  Affinity pts : {results['total_affinity']:6d}")
    print(f"  LLM calls    : {results['llm_calls']:6d}  ({results['elapsed_time_s']:.1f}s)")

    cpu_utils = [m.cpu_util for m in state.machines]
    powers    = [m.current_power() for m in state.machines]
    print(f"  CPU util     : min={np.min(cpu_utils):.1%}  "
          f"mean={np.mean(cpu_utils):.1%}  max={np.max(cpu_utils):.1%}")
    print(f"  Power per m  : min={np.min(powers):.1f}W  "
          f"mean={np.mean(powers):.1f}W  total={np.sum(powers):.1f}W")

    aff_scores = [p.affinity for p in results["history"]]
    if aff_scores:
        affinity_rate = 100 * sum(aff_scores) / len(aff_scores)
        print(f"  Affinity rate: {affinity_rate:.1f}%  "
              f"({sum(aff_scores)}/{len(aff_scores)} placements)")

    print()
    print(f"Results saved to: {output_dir}/")
    print("  decisions.csv  placements.csv  step_stats.csv  llm_stats.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
