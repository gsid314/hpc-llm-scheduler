#!/usr/bin/env python3
"""
Demo script for HPC Multi-Agent Scheduler
Runs on a small synthetic dataset for quick testing.

Usage examples:
  python demo_scheduler.py                        # defaults: 8 apps, 5 machines
  python demo_scheduler.py --num-apps 20 --num-machines 10
  python demo_scheduler.py --num-apps 50 --num-machines 30 --seed 99
  python demo_scheduler.py --model llama3.1 --max-iter 20
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from hpc_scheduler_agents import (
    HPCSchedulerOrchestrator,
    generate_random_state,
)


def main():
    parser = argparse.ArgumentParser(
        description="HPC Multi-Agent Scheduler — Demo (synthetic data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--num-apps",     type=int,   default=8,    help="Number of apps (default: 8)")
    parser.add_argument("--num-machines", type=int,   default=5,    help="Number of machines (default: 5)")
    parser.add_argument("--seed",         type=int,   default=42,   help="RNG seed (default: 42)")
    parser.add_argument("--lambda",       dest="lambda_weight",
                                          type=float, default=0.5,  help="Lambda weight (default: 0.5)")
    parser.add_argument("--max-iter",     type=int,   default=30,   help="Max iterations (default: 30)")
    parser.add_argument("--model",        default="llama3",          help="Ollama model (default: llama3)")
    parser.add_argument("--quiet",        action="store_true",       help="Suppress verbose output")
    args = parser.parse_args()

    print("=" * 80)
    print("HPC Multi-Agent Scheduler — Demo")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Apps:       {args.num_apps}")
    print(f"  Machines:   {args.num_machines}")
    print(f"  Seed:       {args.seed}")
    print(f"  Model:      {args.model}")
    print(f"  Lambda:     {args.lambda_weight}")
    print(f"  Max iter:   {args.max_iter}")
    print()

    state = generate_random_state(
        num_apps=args.num_apps,
        num_machines=args.num_machines,
        seed=args.seed,
    )

    total_instances = sum(app.instances_total for app in state.apps)
    print(f"Generated: {len(state.apps)} apps, {len(state.machines)} machines, "
          f"{total_instances} total instances")
    print()

    orchestrator = HPCSchedulerOrchestrator(
        state=state,
        lambda_weight=args.lambda_weight,
        max_iterations=args.max_iter,
        verbose=not args.quiet,
        model=args.model,
    )

    results = orchestrator.run()

    print()
    print("=" * 80)
    print("Final Results Summary")
    print("=" * 80)
    print(f"Placements made:     {results['total_placements']}/{total_instances}")
    print(f"Total cluster power: {results['total_power_W']:.2f} W")
    print(f"Affinity score:      {results['total_affinity']}")
    print(f"Avg CPU utilization: {results['avg_cpu_util']:.1%}")
    print(f"Time elapsed:        {results['elapsed_time_s']:.2f} s")
    print()

    print("Per-Machine Statistics:")
    print("-" * 80)
    print(f"{'Machine':>8} {'CPU Util':>10} {'Power (W)':>12} {'Instances':>10}")
    print("-" * 80)
    for m in state.machines:
        instances = sum(1 for p in state.history if p.machine_id == m.machine_id)
        print(f"{m.machine_id:8d} {m.cpu_util:10.1%} {m.current_power():12.2f} {instances:10d}")
    print()

    if results["total_placements"] > 0:
        show = min(10, results["total_placements"])
        print(f"Placement Details (last {show}):")
        print("-" * 80)
        for p in results["history"][-show:]:
            print(f"  app_{p.app_id} -> machine_{p.machine_id} | "
                  f"aff={p.affinity}, dP={p.delta_power:6.2f}W, "
                  f"util_after={p.cpu_util_after:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
