"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEST_SCRIPT = BASE_DIR / "tests" / "run_tests.py"
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "benchmark_performance.py"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def run_tests():
    if not TEST_SCRIPT.exists():
        print(f"{RED}Error: Test script not found at {TEST_SCRIPT}{RESET}")
        sys.exit(1)

    cmd = [sys.executable, str(TEST_SCRIPT)] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

def run_benchmarks():
    if not BENCHMARK_SCRIPT.exists():
        print(f"{RED}Error: Benchmark script not found at {BENCHMARK_SCRIPT}{RESET}")
        sys.exit(1)

    cmd = [sys.executable, str(BENCHMARK_SCRIPT)] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
