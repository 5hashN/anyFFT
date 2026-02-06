# Changelog

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
