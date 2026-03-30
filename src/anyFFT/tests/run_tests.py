"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

import subprocess
import sys
import shutil
import argparse
from pathlib import Path

try:
    import anyFFT
except ImportError:
    print("Error: Could not import 'anyFFT'.")
    sys.exit(1)

TEST_DIR = Path(__file__).resolve().parent
BASE_DIR = TEST_DIR.parent
LOCAL_TEST_DIR = TEST_DIR / "local"
DIST_TEST_DIR = TEST_DIR / "dist"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def print_status(name, status, message=""):
    if status == "PASS":
        print(f"{name:<40} {GREEN}[PASS]{RESET} {message}")
    elif status == "FAIL":
        print(f"{name:<40} {RED}[FAIL]{RESET} {message}")
    elif status == "SKIP":
        print(f"{name:<40} {YELLOW}[SKIP]{RESET} {message}")

def run_command(cmd, test_name):
    try:
        print(f"\nRunning {test_name}")
        subprocess.run(cmd, check=True, text=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{RED}Error running {test_name}: Exit Code {e.returncode}{RESET}")
        return False
    except FileNotFoundError:
        print(f"\n{RED}Executable not found for command: {cmd[0]}{RESET}")
        return False

def format_backend(name, tag, is_installed):
    color = GREEN if is_installed else RED
    mark = "x" if is_installed else " "
    return f"{color} [{mark}] {name:<14} {YELLOW}({tag}){RESET}"

def main():
    parser = argparse.ArgumentParser(description="anyFFT Test Runner")
    parser.add_argument(
        "filters",
        nargs="*",
        help="Optional: Filter tests by backend (e.g., 'fftw', 'gpufft') or category ('local', 'dist')."
    )
    args = parser.parse_args()

    filters = [f.lower() for f in args.filters]

    def should_run(script_name, category):
        if not filters:
            return True
        for f in filters:
            if f in script_name.lower() or f == category.lower():
                return True
        return False

    print(f"{CYAN}{'='*60}\nanyFFT Test Runner\n{'='*60}{RESET}")

    if filters:
        print(f"{YELLOW}Active Filters: {', '.join(filters)}{RESET}\n")

    backends = {
        "fftw":        ["FFTW",         "Local",       anyFFT.has_backend("fftw")],
        "fftw_mpi":    ["FFTW-MPI",     "Distributed", anyFFT.has_backend("fftw_mpi")],
        "cufft":       ["cuFFT",        "Local",       anyFFT.has_backend("cufft")],
        "hipfft":      ["hipFFT",       "Local",       anyFFT.has_backend("hipfft")],
        "cufftmp":     ["cuFFTMp",      "Distributed", anyFFT.has_backend("cufftmp")],
        "cufft_dist":  ["cuFFT_dist",   "Distributed", anyFFT.has_backend("cufft_dist")],
        "hipfft_dist": ["hipFFT_dist",  "Distributed", anyFFT.has_backend("hipfft_dist")],
    }

    has_mpiexec = shutil.which("mpiexec") is not None or shutil.which("mpirun") is not None

    print(f"{CYAN}Installed Backends:{RESET}")
    for key, (name, tag, is_installed) in backends.items():
        print(format_backend(name, tag, is_installed))
    print(f"{CYAN}{'-' * 60}{RESET}")

    results = {"pass": 0, "fail": 0, "skip": 0}
    failed_tests = []

    local_tests = sorted(LOCAL_TEST_DIR.glob("test_*.py")) if LOCAL_TEST_DIR.exists() else []

    print(f"\n{CYAN}{'='*60}\n[{'LOCAL TESTS':^20}]\n{'='*60}{RESET}")

    if not local_tests:
        print("No local tests found.")

    for path in local_tests:
        script = path.name

        if not should_run(script, "local"):
            continue

        skip_reason = None

        if "fftw" in script and not backends["fftw"][2]:
            skip_reason = "(FFTW backend not installed)"
        elif "gpufft" in script and not (backends["cufft"][2] or backends["hipfft"][2]):
            skip_reason = "(Unified gpuFFT backend not installed)"

        if skip_reason:
            print_status(script, "SKIP", skip_reason)
            results["skip"] += 1
            continue

        success = run_command([sys.executable, str(path)], script)
        if success:
            print_status(script, "PASS")
            results["pass"] += 1
        else:
            print_status(script, "FAIL")
            results["fail"] += 1
            failed_tests.append(script)

    dist_tests = sorted(DIST_TEST_DIR.glob("test_*.py")) if DIST_TEST_DIR.exists() else []

    print(f"\n{CYAN}{'='*60}\n[{'DISTRIBUTED TESTS':^20}]\n{'='*60}{RESET}")

    if not has_mpiexec and dist_tests:
        print(f"{YELLOW}Skipping all Distributed tests (mpirun/mpiexec not found on system){RESET}")
        results["skip"] += len(dist_tests)
    elif not dist_tests:
        print("No distributed tests found.")
    else:
        for path in dist_tests:
            script = path.name

            if not should_run(script, "dist"):
                continue

            skip_reason = None

            if "fftw" in script and not backends["fftw_mpi"][2]:
                skip_reason = "(FFTW-MPI backend not installed)"
            elif "gpufft_dist" in script and not (backends["cufft_dist"][2] or backends["hipfft_dist"][2]):
                skip_reason = "(Unified gpuFFT-dist backend not installed)"
            elif "cufftmp" in script and not backends["cufftmp"][2]:
                skip_reason = "(cuFFTMp backend not installed)"

            if skip_reason:
                print_status(script, "SKIP", skip_reason)
                results["skip"] += 1
                continue

            cmd = ["mpirun", "-n", "2", sys.executable, str(path)]
            success = run_command(cmd, script)

            if success:
                print_status(script, "PASS")
                results["pass"] += 1
            else:
                print_status(script, "FAIL")
                results["fail"] += 1
                failed_tests.append(script)

    print(f"\n{CYAN}{'='*60}\nFINAL SUMMARY\n{'='*60}{RESET}")
    print(f"Passed:  {GREEN}{results['pass']}{RESET}")
    print(f"Skipped: {YELLOW}{results['skip']}{RESET}")
    print(f"Failed:  {RED}{results['fail']}{RESET}")

    if failed_tests:
        print("\nFailures:")
        for t in failed_tests:
            print(f"  - {t}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}ALL CHECKS PASSED.{RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()
