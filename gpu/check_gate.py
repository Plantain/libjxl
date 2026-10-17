#!/usr/bin/env python3
"""Check saved WP3 numerical, timing and memory evidence; emit a small summary."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')


def read(name):
    return json.loads((root / name).read_text())


numeric = read('numerics-d1.2-b32.json')
timings = [read(f'timing-b{b}.json') for b in (4, 8, 16, 32)]
memory = [read(f'memory-b{b}.json') for b in (4, 8, 16, 32)]
transforms = read('transforms.json')
profile = read('nsys.json')
passing = [t for t in timings if t['batch'] >= 8
           and t['evaluation_ms_per_tile'] <= .4
           and t['setup_ms_per_tile'] <= .15]
checks = {
    'numerical': (numeric['tiles'] == 120
                  and numeric['linear_mae'] <= 1e-4
                  and numeric['diff_relative_p99'] <= 1e-3
                  and numeric['threshold_1_agreement'] >= .995
                  and numeric['update_threshold_agreement'] >= .995
                  and numeric['tile_argmax_agreement'] >= .99),
    'transforms': (transforms['pass']
                   and {c['strategy'] for c in transforms['transform_cases']}
                   == set(range(21))),
    'timing': bool(passing),
    'memory': all(m['samples'] > 0 and m['peak_process_vram_mib'] <= 2048
                  for m in memory),
    'profile_collected': bool(profile['kernels']),
}
best = min(passing, key=lambda t: t['evaluation_ms_per_tile']) if passing else None
result = {
    'gate_pass': all(checks.values()),
    'checks': checks,
    'best_passing_timing': best,
    'maximum_process_vram_mib': max(m['peak_process_vram_mib'] for m in memory),
    'linear_mae': numeric['linear_mae'],
    'diff_relative_p99': numeric['diff_relative_p99'],
    'threshold_1_agreement': numeric['threshold_1_agreement'],
    'tile_argmax_agreement': numeric['tile_argmax_agreement'],
}
print(json.dumps(result, indent=2))
sys.exit(0 if result['gate_pass'] else 1)
