"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
TEST_SCRIPT = BASE_DIR / "tests" / "run_tests.py"
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "benchmark_performance.py"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

class VerboseFileLogger:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def pytest_sessionstart(self, session):
        self.file = open(self.filename, 'w', encoding='utf-8')
        self.file.write(f"anyFFT Verbose Test Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write("="*80 + "\n\n")

    def pytest_runtest_logreport(self, report):
        if report.when == 'setup' and report.outcome == 'skipped':
            reason = str(report.longrepr).split('\n')[-1] if report.longrepr else "Unknown"
            self.file.write(f"SKIPPED | {report.nodeid} \n          Reason: {reason}\n")

        elif report.when == 'call':
            status = report.outcome.upper()
            duration = f"{report.duration:.3f}s"
            self.file.write(f"{status:<7} | {report.nodeid} ({duration})\n")

        if report.failed:
            self.file.write(f"\n{'-'*80}\nERROR DETAILS:\n{report.longrepr}\n{'-'*80}\n\n")

    def pytest_sessionfinish(self, session, exitstatus):
        self.file.write(f"\n{'='*80}\n")
        self.file.write(f"Session finished with Exit Code: {exitstatus}\n")
        if self.file:
            self.file.close()

def run_tests():
    import pytest

    is_mpi = "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ
    is_main_rank = True

    if is_mpi:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() > 0:
            is_main_rank = False
            f = open(os.devnull, 'w')
            sys.stdout = f
            sys.stderr = f

    raw_args = sys.argv[1:]
    args = [arg for arg in raw_args if arg not in ("-v", "--verbose")]
    has_user_args = bool(args)

    if not any(arg.endswith(".py") for arg in args) and "--pyargs" not in args:
        args.extend(["--pyargs", "anyFFT.tests"])

    if is_mpi:
        if "--with-mpi" not in args:
            args.append("--with-mpi")
        if not any(a.startswith("-m") for a in args):
            args.extend(["-m", "mpi"])
    elif not has_user_args:
        args.extend(["-m", "cpu and not mpi"])

    plugins = []
    if is_main_rank:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(os.getcwd(), f"anyfft_test_run_{timestamp}.log")

        print(f"{CYAN}Verbose Log saved to:    {log_filename}\n{RESET}")

        plugins.append(VerboseFileLogger(log_filename))

    sys.exit(pytest.main(args, plugins=plugins))

def run_benchmarks():
    if not BENCHMARK_SCRIPT.exists():
        print(f"{RED}Error: Benchmark script not found at {BENCHMARK_SCRIPT}{RESET}")
        sys.exit(1)

    cmd = [sys.executable, str(BENCHMARK_SCRIPT)] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
