import subprocess
import sys
import os
import shutil

try:
    import anyFFT
except ImportError:
    print("\033[91mError: Could not import 'anyFFT'. Make sure it is installed.\033[0m")
    sys.exit(1)

TEST_DIR = "tests"
SERIAL_DIR = os.path.join(TEST_DIR, "serial")
MPI_DIR = os.path.join(TEST_DIR, "mpi")

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


def main():
    print(f"{'='*60}\nanyFFT Test Runner\n{'='*60}")

    # Backend Detection via anyFFT
    has_fftw = anyFFT.has_backend("fftw")
    has_fftw_mpi = anyFFT.has_backend("fftw_mpi")
    has_cuda = anyFFT.has_backend("cufft")
    has_cuda_mpi = anyFFT.has_backend("cufft_mpi")

    # Check for system MPI runner
    has_mpiexec = shutil.which("mpiexec") is not None or shutil.which("mpirun") is not None

    print(f"Installed Backends:")
    print(f"  FFTW (Serial):   {'Yes' if has_fftw else 'No'}")
    print(f"  FFTW (MPI):      {'Yes' if has_fftw_mpi else 'No'}")
    print(f"  cuFFT (Serial):  {'Yes' if has_cuda else 'No'}")
    print(f"  cuFFT (MPI):     {'Yes' if has_cuda_mpi else 'No'}")
    print("-" * 60)

    results = {"pass": 0, "fail": 0, "skip": 0}
    failed_tests = []

    # Serial Tests
    if os.path.exists(SERIAL_DIR):
        serial_tests = sorted([f for f in os.listdir(SERIAL_DIR) if f.startswith("test_") and f.endswith(".py")])
    else:
        serial_tests = []

    print(f"[{'SERIAL TESTS':^20}]")
    if not serial_tests:
        print("No serial tests found.")

    for script in serial_tests:
        path = os.path.join(SERIAL_DIR, script)
        skip_reason = None

        # Determine if we should skip based on filename vs installed backend
        if "fftw" in script and not has_fftw:
            skip_reason = "(FFTW backend not installed)"
        elif "cufft" in script and not has_cuda:
            skip_reason = "(cuFFT backend not installed)"

        if skip_reason:
            print_status(script, "SKIP", skip_reason)
            results["skip"] += 1
            continue

        # Run Test
        success = run_command([sys.executable, path], script)
        if success:
            print_status(script, "PASS")
            results["pass"] += 1
        else:
            print_status(script, "FAIL")
            results["fail"] += 1
            failed_tests.append(script)

    # MPI Tests
    if os.path.exists(MPI_DIR):
        mpi_tests = sorted([f for f in os.listdir(MPI_DIR) if f.startswith("test_") and f.endswith(".py")])
    else:
        mpi_tests = []

    print(f"\n[{'MPI TESTS':^20}]")

    # Global check: if no mpi runner, skip all
    if not has_mpiexec and mpi_tests:
         print(f"{YELLOW}Skipping all MPI tests (mpirun/mpiexec not found on system){RESET}")
         results["skip"] += len(mpi_tests)
    elif not mpi_tests:
        print("No MPI tests found.")
    else:
        for script in mpi_tests:
            path = os.path.join(MPI_DIR, script)
            skip_reason = None

            # Determine if we should skip based on filename vs installed backend
            if "fftw" in script and not has_fftw_mpi:
                skip_reason = "(FFTW-MPI backend not installed)"
            elif "cufft" in script and not has_cuda_mpi:
                skip_reason = "(cuFFT-MPI backend not installed)"

            if skip_reason:
                print_status(script, "SKIP", skip_reason)
                results["skip"] += 1
                continue

            # Run MPI Test
            # Default to 2 processes for testing
            cmd = ["mpirun", "-n", "2", sys.executable, path]
            success = run_command(cmd, script)

            if success:
                print_status(script, "PASS")
                results["pass"] += 1
            else:
                print_status(script, "FAIL")
                results["fail"] += 1
                failed_tests.append(script)

    # Summary
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


if __name__ == "__main__":
    main()