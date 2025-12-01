"""
Example: Simple shmem4py program for testing the frontend.

This is a mock shmem4py program that the frontend should be able to parse.
It performs a simple nearest-neighbor communication pattern (halo exchange).

Usage (once frontend is implemented):
    python ../../python/shmem4py_mlir/cli.py halo_step_simple.py --fn halo_step --emit-mlir ir.mlir
"""

import numpy as np
# Simulated shmem4py API (using standard shmem_* naming)


def shmem_init():
    """Initialize OpenSHMEM runtime."""
    pass


def shmem_finalize():
    """Finalize OpenSHMEM runtime."""
    pass


def shmem_barrier_all():
    """Global synchronization barrier."""
    pass


def shmem_my_pe() -> int:
    """Get this PE's rank."""
    return 0


def shmem_n_pes() -> int:
    """Get number of PEs."""
    return 1


def shmem_put(src: np.ndarray, dest_pe: int, dest_offset: int):
    """
    Put data to remote PE.
    
    Args:
        src: Local array to send
        dest_pe: Destination PE rank
        dest_offset: Offset in remote array
    """
    pass


def shmem_get(dest: np.ndarray, src_pe: int, src_offset: int, count: int):
    """
    Get data from remote PE.
    
    Args:
        dest: Local array to receive into
        src_pe: Source PE rank
        src_offset: Offset in remote array
        count: Number of elements
    """
    pass


def halo_step_simple(u: np.ndarray, v: np.ndarray, niters: int):
    """
    Simple halo exchange (nearest-neighbor communication).
    
    Exchanges boundary values with left and right neighbors in a 1D domain.
    This is a minimal example for testing the frontend.
    
    Args:
        u: Distributed 1D array
        v: Temporary array for communication
        niters: Number of iterations
    """
    shmem_init()
    
    me = shmem_my_pe()
    npes = shmem_n_pes()
    n = len(u)
    
    # Boundary indices
    left_neighbor = (me - 1) % npes
    right_neighbor = (me + 1) % npes
    
    for iteration in range(niters):
        # Interior computation (simple copy for now)
        for i in range(1, n - 1):
            v[i] = u[i]
        
        # Synchronize before communication
        shmem_barrier_all()
        
        # Send boundaries to neighbors
        shmem_put(u[0:1], left_neighbor, n - 1)
        shmem_put(u[n - 1:n], right_neighbor, 0)
        
        # Synchronize after communication
        shmem_barrier_all()
        
        # Receive boundaries from neighbors
        shmem_get(v[0:1], right_neighbor, 0, 1)
        shmem_get(v[n - 1:n], left_neighbor, n - 1, 1)
    
    shmem_finalize()


def simple_barrier_test(num_iters: int):
    """
    Minimal test: just barriers and rank queries.
    
    Args:
        num_iters: Number of barrier iterations
    """
    shmem_init()
    
    me = shmem_my_pe()
    npes = shmem_n_pes()
    
    for i in range(num_iters):
        shmem_barrier_all()
    
    shmem_finalize()


if __name__ == "__main__":
    # For testing, just run the minimal example
    print("This module contains shmem4py example code for frontend testing.")
    print("It is not meant to be executed directly.")
    simple_barrier_test(1)
