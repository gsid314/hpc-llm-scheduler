#!/usr/bin/env python3
"""
Analyze HPC Scheduler Results
Provides detailed analysis and visualization of scheduling outcomes
"""

import json
import sys
from collections import defaultdict


def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_placements(results):
    """Analyze placement statistics"""
    placements = results.get('placements', [])
    
    if not placements:
        print("No placements found in results")
        return
    
    print("=" * 80)
    print("PLACEMENT ANALYSIS")
    print("=" * 80)
    print()
    
    # Basic stats
    total = len(placements)
    affinity_count = sum(1 for p in placements if p['affinity'] == 1)
    affinity_rate = 100 * affinity_count / total if total > 0 else 0
    
    print(f"Total placements:          {total:6d}")
    print(f"Affinity=1 placements:     {affinity_count:6d} ({affinity_rate:.1f}%)")
    print(f"Affinity=0 placements:     {total - affinity_count:6d}")
    print()
    
    # Power distribution
    powers = [p['delta_power'] for p in placements]
    if powers:
        import numpy as np
        print("Incremental Power (ΔP):")
        print(f"  Min:        {np.min(powers):8.2f} W")
        print(f"  Max:        {np.max(powers):8.2f} W")
        print(f"  Mean:       {np.mean(powers):8.2f} W")
        print(f"  Median:     {np.median(powers):8.2f} W")
        print(f"  Std Dev:    {np.std(powers):8.2f} W")
        print()
    
    # CPU utilization distribution
    utils = [p['cpu_util_after'] for p in placements]
    if utils:
        import numpy as np
        print("CPU Utilization After Placement:")
        print(f"  Min:        {np.min(utils):8.1%}")
        print(f"  Max:        {np.max(utils):8.1%}")
        print(f"  Mean:       {np.mean(utils):8.1%}")
        print(f"  Median:     {np.median(utils):8.1%}")
        print()


def analyze_per_app(results):
    """Analyze placements per application"""
    placements = results.get('placements', [])
    
    if not placements:
        return
    
    print("=" * 80)
    print("PER-APPLICATION STATISTICS")
    print("=" * 80)
    print()
    
    # Group by app
    app_stats = defaultdict(lambda: {
        'count': 0,
        'affinity_1': 0,
        'total_power': 0,
        'machines': set()
    })
    
    for p in placements:
        app_id = p['app_id']
        app_stats[app_id]['count'] += 1
        app_stats[app_id]['affinity_1'] += p['affinity']
        app_stats[app_id]['total_power'] += p['delta_power']
        app_stats[app_id]['machines'].add(p['machine_id'])
    
    # Sort by app_id
    print(f"{'App ID':>8} {'Instances':>10} {'Aff=1':>8} {'Machines':>10} {'Avg ΔP (W)':>12}")
    print("-" * 80)
    
    for app_id in sorted(app_stats.keys())[:50]:  # Show first 50
        stats = app_stats[app_id]
        avg_power = stats['total_power'] / stats['count']
        print(f"{app_id:8d} {stats['count']:10d} {stats['affinity_1']:8d} "
              f"{len(stats['machines']):10d} {avg_power:12.2f}")
    
    if len(app_stats) > 50:
        print(f"... ({len(app_stats) - 50} more apps)")
    
    print()


def analyze_per_machine(results):
    """Analyze placements per machine"""
    placements = results.get('placements', [])
    
    if not placements:
        return
    
    print("=" * 80)
    print("PER-MACHINE STATISTICS")
    print("=" * 80)
    print()
    
    # Group by machine
    machine_stats = defaultdict(lambda: {
        'count': 0,
        'affinity_1': 0,
        'total_power': 0,
        'apps': set(),
        'final_util': 0
    })
    
    for p in placements:
        machine_id = p['machine_id']
        machine_stats[machine_id]['count'] += 1
        machine_stats[machine_id]['affinity_1'] += p['affinity']
        machine_stats[machine_id]['total_power'] += p['delta_power']
        machine_stats[machine_id]['apps'].add(p['app_id'])
        machine_stats[machine_id]['final_util'] = p['cpu_util_after']  # Last placement
    
    # Find most/least loaded
    by_count = sorted(machine_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print("Top 10 Most Loaded Machines:")
    print(f"{'Machine':>8} {'Instances':>10} {'Aff=1':>8} {'Apps':>8} {'Final Util':>12} {'Total ΔP':>12}")
    print("-" * 80)
    
    for machine_id, stats in by_count[:10]:
        print(f"{machine_id:8d} {stats['count']:10d} {stats['affinity_1']:8d} "
              f"{len(stats['apps']):8d} {stats['final_util']:11.1%} "
              f"{stats['total_power']:11.2f}W")
    
    print()
    
    # Machines with zero placements
    config = results.get('configuration', {})
    total_machines = config.get('total_machines', 0)
    used_machines = len(machine_stats)
    unused_machines = total_machines - used_machines
    
    print(f"Machines used:    {used_machines:6d} / {total_machines}")
    print(f"Machines unused:  {unused_machines:6d}")
    print()


def analyze_temporal_patterns(results):
    """Analyze how metrics change over time"""
    placements = results.get('placements', [])
    
    if not placements or len(placements) < 10:
        return
    
    print("=" * 80)
    print("TEMPORAL PATTERNS")
    print("=" * 80)
    print()
    
    import numpy as np
    
    # Divide into quartiles
    n = len(placements)
    q1 = placements[:n//4]
    q2 = placements[n//4:n//2]
    q3 = placements[n//2:3*n//4]
    q4 = placements[3*n//4:]
    
    quartiles = [
        ("First 25%", q1),
        ("Second 25%", q2),
        ("Third 25%", q3),
        ("Last 25%", q4)
    ]
    
    print(f"{'Period':>12} {'Aff=1 Rate':>12} {'Avg ΔP (W)':>12} {'Avg Util':>12}")
    print("-" * 80)
    
    for name, quarter in quartiles:
        if not quarter:
            continue
        
        aff_rate = 100 * sum(p['affinity'] for p in quarter) / len(quarter)
        avg_power = np.mean([p['delta_power'] for p in quarter])
        avg_util = np.mean([p['cpu_util_after'] for p in quarter])
        
        print(f"{name:>12} {aff_rate:11.1f}% {avg_power:11.2f}W {avg_util:11.1%}")
    
    print()
    print("Observations:")
    
    # Check if affinity rate decreases over time
    aff_rates = [100 * sum(p['affinity'] for p in q) / len(q) for _, q in quartiles if q]
    if aff_rates[0] > aff_rates[-1] + 5:
        print("  - Affinity rate decreased over time (harder to satisfy preferences later)")
    elif aff_rates[-1] > aff_rates[0] + 5:
        print("  - Affinity rate increased over time (unusual - check scheduling logic)")
    
    # Check if power increases over time
    power_vals = [np.mean([p['delta_power'] for p in q]) for _, q in quartiles if q]
    if power_vals[-1] > power_vals[0] * 1.2:
        print("  - Incremental power increased significantly (machines filling up)")
    
    print()


def generate_summary(results):
    """Generate executive summary"""
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print()
    
    config = results.get('configuration', {})
    res = results.get('results', {})
    
    # Completion
    total_instances = config.get('total_instances', 0)
    placements = res.get('total_placements', 0)
    completion = res.get('completion_rate', 0)
    
    print(f"Scheduling Task:")
    print(f"  {config.get('total_apps', 0)} applications")
    print(f"  {total_instances} instances to place")
    print(f"  {config.get('total_machines', 0)} machines available")
    print(f"  λ={config.get('lambda_weight', 0.5)} (power-affinity balance)")
    print()
    
    print(f"Outcomes:")
    print(f"  {placements} instances placed ({completion:.1f}% complete)")
    print(f"  {res.get('total_affinity', 0)} affinity points earned")
    print(f"  {res.get('total_power_W', 0):.1f}W total cluster power")
    print(f"  {res.get('avg_cpu_util', 0):.1%} average CPU utilization")
    print()
    
    print(f"Performance:")
    print(f"  {res.get('elapsed_time_s', 0):.1f}s execution time")
    
    if placements > 0:
        time_per_placement = res.get('elapsed_time_s', 0) / placements
        print(f"  {time_per_placement:.2f}s per placement")
    
    print()
    
    # Grade the results
    print("Assessment:")
    
    if completion >= 99:
        print("  ✓ Excellent completion rate")
    elif completion >= 90:
        print("  ✓ Good completion rate")
    elif completion >= 70:
        print("  ⚠ Moderate completion rate - some constraints may be too restrictive")
    else:
        print("  ✗ Low completion rate - check data quality and constraints")
    
    if placements > 0:
        placements_list = results.get('placements', [])
        aff_rate = 100 * sum(p['affinity'] for p in placements_list) / len(placements_list)
        
        if aff_rate >= 70:
            print("  ✓ High affinity satisfaction rate")
        elif aff_rate >= 50:
            print("  ✓ Moderate affinity satisfaction")
        else:
            print("  ⚠ Low affinity satisfaction - power may be over-weighted")
    
    machine_stats = results.get('machine_stats', {})
    if machine_stats:
        cpu_mean = machine_stats.get('cpu_util_mean', 0)
        if cpu_mean < 0.3:
            print("  ⚠ Low average utilization - cluster underutilized")
        elif cpu_mean > 0.8:
            print("  ⚠ High average utilization - may risk overload")
        else:
            print("  ✓ Balanced utilization")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results.json>")
        print()
        print("Example:")
        print("  python analyze_results.py scheduler_results.json")
        return 1
    
    filepath = sys.argv[1]
    
    try:
        results = load_results(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return 1
    except json.JSONDecodeError:
        print(f"Error: '{filepath}' is not valid JSON")
        return 1
    
    print()
    generate_summary(results)
    analyze_placements(results)
    analyze_per_app(results)
    analyze_per_machine(results)
    analyze_temporal_patterns(results)
    
    print("=" * 80)
    print("Analysis complete")
    print("=" * 80)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())