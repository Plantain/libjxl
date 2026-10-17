#!/usr/bin/env python3
"""Measure this work package's process VRAM, including CUDA runtime overhead."""
import json
import os
import signal
import subprocess
import sys
import time

running = True
peak = 0
samples = 0

def stop(*_):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
while running:
    text = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    total = 0
    found = False
    for line in text.splitlines():
        pid, memory = (part.strip() for part in line.split(','))
        try:
            with open('/proc/' + pid + '/cmdline', 'rb') as f:
                command = f.read()
            if b'/w/wp3/build/gpu/wp3_spike' in command:
                total += int(memory)
                found = True
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass
    if found:
        samples += 1
        peak = max(peak, total)
    time.sleep(.05)
with open(sys.argv[1], 'w') as f:
    json.dump({'peak_process_vram_mib': peak, 'samples': samples,
               'within_2048_mib': bool(samples and peak <= 2048)}, f, indent=2)
    f.write('\n')
