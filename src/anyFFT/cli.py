import subprocess
import sys
import shutil
from pathlib import Path

try:
    import anyFFT
except ImportError:
    print("\033[91mError: Could not import 'anyFFT'. Make sure it is installed.\033[0m")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "tests"
SERIAL_DIR = TEST_DIR / "serial"
MPI_DIR = TEST_DIR / "mpi"
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "benchmark_performance.py"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
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

def run_tests():
    print(f"{'='*60}\nanyFFT Test Runner\n{'='*60}")

    has_fftw = anyFFT.has_backend("fftw")
    has_fftw_mpi = anyFFT.has_backend("fftw_mpi")
    has_cuda = anyFFT.has_backend("cufft")
    has_cuda_mpi = anyFFT.has_backend("cufft_mpi")

    has_mpiexec = shutil.which("mpiexec") is not None or shutil.which("mpirun") is not None

    print("Installed Backends:")
    print(f"  FFTW (Serial):   {'Yes' if has_fftw else 'No'}")
    print(f"  FFTW (MPI):      {'Yes' if has_fftw_mpi else 'No'}")
    print(f"  cuFFT (Serial):  {'Yes' if has_cuda else 'No'}")
    print(f"  cuFFT (MPI):     {'Yes' if has_cuda_mpi else 'No'}")
    print("-" * 60)

    results = {"pass": 0, "fail": 0, "skip": 0}
    failed_tests = []

    serial_tests = sorted(SERIAL_DIR.glob("test_*.py")) if SERIAL_DIR.exists() else []

    print(f"[{'SERIAL TESTS':^20}]")
    if not serial_tests:
        print("No serial tests found.")

    for path in serial_tests:
        script = path.name
        skip_reason = None

        if "fftw" in script and not has_fftw:
            skip_reason = "(FFTW backend not installed)"
        elif "cufft" in script and not has_cuda:
            skip_reason = "(cuFFT backend not installed)"

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

    mpi_tests = sorted(MPI_DIR.glob("test_*.py")) if MPI_DIR.exists() else []

    print(f"\n[{'MPI TESTS':^20}]")

    if not has_mpiexec and mpi_tests:
         print(f"{YELLOW}Skipping all MPI tests (mpirun/mpiexec not found on system){RESET}")
         results["skip"] += len(mpi_tests)
    elif not mpi_tests:
        print("No MPI tests found.")
    else:
        for path in mpi_tests:
            script = path.name
            skip_reason = None

            if "fftw" in script and not has_fftw_mpi:
                skip_reason = "(FFTW-MPI backend not installed)"
            elif "cufft" in script and not has_cuda_mpi:
                skip_reason = "(cuFFT-MPI backend not installed)"

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

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
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

def run_benchmarks():
    if not BENCHMARK_SCRIPT.exists():
        print(f"{RED}Error: Benchmark script not found at {BENCHMARK_SCRIPT}{RESET}")
        sys.exit(1)

    cmd = [sys.executable, str(BENCHMARK_SCRIPT)] + sys.argv[1:]
    print(f"\nRunning {BENCHMARK_SCRIPT.name}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
