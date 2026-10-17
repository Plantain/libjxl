#!/usr/bin/env python3
"""Summarize the private WP3 dump format without numpy or GPU access."""
import glob
import json
from pathlib import Path
import re
import struct
import sys
N, B = 512 * 512, 64 * 64
counts = [0] * 27
filters = set()
for path in sorted(glob.glob(sys.argv[1])):
    with open(path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        assert (magic, version) == (0x3350574a, 1)
        params = f.read(224)
        filters.add(tuple(struct.unpack_from('<II', params, 200)))
        f.seek(8 + 224 + (3 * N + 3 * B + B) * 4)
        kinds, first = f.read(B), f.read(B)
        for kind, anchor in zip(kinds, first):
            if anchor:
                counts[kind] += 1
header = (Path(__file__).resolve().parent.parent / 'lib/jxl/ac_strategy.h').read_text()
names = {int(v): n for n, v in re.findall(r'^  (\w+) = (\d+),?$', header, re.M)}
print(json.dumps({'files': len(glob.glob(sys.argv[1])), 'gab_epf': sorted(filters),
                  'strategy_counts': counts,
                  'present_types': sum(v > 0 for v in counts),
                  'transform_anchors': sum(counts),
                  'strategies': [{'type': i, 'name': names[i], 'count': v,
                                  'implemented': i <= 20}
                                 for i, v in enumerate(counts)]}, indent=2))
