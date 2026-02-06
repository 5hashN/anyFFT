import subprocess
import sys
import os
import shutil

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
        print(f"\n--- Running {test_name} ---")
        subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{RED}Error running {test_name}: Exit Code {e.returncode}{RESET}")
        return False
    except FileNotFoundError:
        print(f"\n{RED}Executable not found for command: {cmd[0]}{RESET}")
        return False

def main():
    print(f"{'='*60}\nanyFFT Test Runner\n{'='*60}")

    has_mpi = shutil.which("mpiexec") is not None
    has_gpu = False
    try:
        import cupy
        if shutil.which("nvcc"): has_gpu = True
    except ImportError: pass

    print(f"Environment: MPI={'Yes' if has_mpi else 'No'} | GPU={'Yes' if has_gpu else 'No'}\n")

    results = {"pass": 0, "fail": 0, "skip": 0}
    failed_tests = []

    serial_tests = sorted([f for f in os.listdir(SERIAL_DIR) if f.startswith("test_") and f.endswith(".py")])
    mpi_tests = sorted([f for f in os.listdir(MPI_DIR) if f.startswith("test_") and f.endswith(".py")])

    print(f"[{'SERIAL TESTS':^20}]")
    for script in serial_tests:
        path = os.path.join(SERIAL_DIR, script)

        if "cufft" in script and not has_gpu:
            print_status(script, "SKIP", "(No GPU found)")
            results["skip"] += 1
            continue

        success = run_command([sys.executable, path], script)
        if success: results["pass"] += 1
        else:
            results["fail"] += 1
            failed_tests.append(script)

    print(f"\n[{'MPI TESTS':^20}]")
    if not has_mpi:
        print(f"{YELLOW}Skipping all MPI tests (mpirun not found){RESET}")
        results["skip"] += len(mpi_tests)
    else:
        for script in mpi_tests:
            path = os.path.join(MPI_DIR, script)

            if "cufft" in script and not has_gpu:
                print_status(script, "SKIP", "(No GPU found)")
                results["skip"] += 1
                continue

            cmd = ["mpirun", "-n", "2", sys.executable, path]
            success = run_command(cmd, script)

            if success: results["pass"] += 1
            else:
                results["fail"] += 1
                failed_tests.append(script)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Passed:  {GREEN}{results['pass']}{RESET}")
    print(f"Skipped: {YELLOW}{results['skip']}{RESET}")
    print(f"Failed:  {RED}{results['fail']}{RESET}")

    if failed_tests:
        print("\nFailures:")
        for t in failed_tests: print(f"  - {t}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}ALL CHECKS PASSED.{RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()