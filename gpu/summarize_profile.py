#!/usr/bin/env python3
"""Convert nsys cuda_gpu_kern_sum CSV into the small WP3 timing evidence."""
import csv
import io
import json
import sys

lines = open(sys.argv[1]).readlines()
start = next(i for i, line in enumerate(lines) if 'Total Time (ns)' in line)
rows = list(csv.DictReader(io.StringIO(''.join(lines[start:]))))
normalizer = int(sys.argv[2]) * int(sys.argv[3])
kernels = []
for row in rows:
    if not row.get('Total Time (ns)'):
        continue
    kernels.append({'name': row['Name'], 'instances': int(row['Instances']),
                    'ms_per_tile': int(row['Total Time (ns)']) / 1e6 / normalizer})
print(json.dumps({'batch': int(sys.argv[2]), 'evaluations': int(sys.argv[3]),
                  'kernel_sum_ms_per_tile': sum(k['ms_per_tile'] for k in kernels),
                  'kernels': kernels}, indent=2))
