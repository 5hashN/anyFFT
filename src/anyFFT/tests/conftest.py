
def pytest_configure(config):
    """
    Register custom markers programmatically so they are available
    even when running tests outside the project root directory.
    """
    config.addinivalue_line("markers", "cpu: marks tests that run purely on the host")
    config.addinivalue_line("markers", "gpu: marks tests that require a CUDA/ROCm device")
    config.addinivalue_line("markers", "mpi: marks tests that require MPI (run with mpiexec)")
