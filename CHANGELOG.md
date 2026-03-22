# Changelog

## [0.1.10] - 2026-03-21

- integrated hipfft to make a unified gpu backend for local transforms
- refactored and organised the structure
- further improvements to the scripts and cli args

## [0.1.9] - 2026-03-17

- moved benchmarking and tests to anyfft cli
- improvements to the scripts and cli args

## [0.1.8] - 2026-02-15

- converted all modules to be as generic as possible
- added checks to the python-facing factory function
- general overall cleanup

## [0.1.7] - 2026-02-07

- fixed github actions workflow for fftw serial module
- updated test script to handle installed backends
- fixed `setup.py` to compile with `nvcc`

## [0.1.6] - 2026-02-07

- added github workflows for testing
- attempt at linting through `black` but giving up for now
- benchmarking performance testing

## [0.1.5] - 2026-02-06

- added multithreading through OpenMP and planner rigor flags for fftw serial module
- basic benchmarking script against some python FFTs

## [0.1.4] - 2026-02-06

- setup GIL release blocks for wrapping `fftw_execute`/`cufftExec` calls
- minor refactoring variable names for homogenising the code
- streamlined testing module through `run_tests.py`

## [0.1.3] - 2026-02-05

- restructuring based on the "src layout" prescribed by PyPI
- added license headers and updated `README.md`

## [0.1.2] - 2026-02-05

- support for transforms along specified axes for serial modules
- restructuring for headers
- updated `README.md` for showing current support and usage samples
- moved `.o` file generation out of `cpp`

## [0.1.1] - 2026-01-29

- preliminary support for parallel mpi modules
- tested for fftw-mpi c2c in-place and out-of-place transforms, and r2c in-place transforms
- `setup.py` improved for providing libraries through environment variables

## [0.1.0] - 2026-01-14

- implementation for r2c, c2r, c2c for fftw and cufft
- implementation for in-place transforms
- testing validated for cpu and gpu

## [0.0.2] - 2026-01-14

- integrated serial binding for fftw and cufft
- modularised for parallel implementations
- license, manifest added
- `setup.py` improved for conditional compiling

## [0.0.1] - 2025-10-06

- initial release
- serial fftw binding for r2c, c2r with single, double precision
- general compliation methods added
