/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include <mpi.h>

// Helper to convert the integer handle passed from Python (mpi4py)
// into a native C++ MPI_Comm object.
inline MPI_Comm get_mpi_comm(int py_comm_handle) {
    return MPI_Comm_f2c((MPI_Fint)py_comm_handle);
}