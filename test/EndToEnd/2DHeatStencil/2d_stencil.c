/*
 * 2D Heat Stencil with OpenSHMEM - Simplified
 *
 * Tests RMA operations (shmem_double_get) without barriers
 */

#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    shmem_init();
    
    int me = shmem_my_pe();
    int npes = shmem_n_pes();
    
    // Allocate symmetric memory
    double *local_data = (double *)shmem_malloc(10 * sizeof(double));
    double *remote_data = (double *)shmem_malloc(10 * sizeof(double));
    
    // Initialize local data
    for (int i = 0; i < 10; i++) {
        local_data[i] = (double)(me * 10 + i);
    }
    
    if (me == 0) {
        printf("2D Heat Stencil Test: %d PEs\n", npes);
        printf("PE 0: Testing shmem_double_get from PE 1\n");
    }
    
    // PE 0 gets data from PE 1 (if PE 1 exists)
    if (me == 0 && npes > 1) {
        shmem_double_get(remote_data, local_data, 10, 1);
        
        printf("PE 0: Retrieved data from PE 1:\n");
        for (int i = 0; i < 10; i++) {
            printf("  remote_data[%d] = %.1f\n", i, remote_data[i]);
        }
    }
    
    if (me == 0) {
        printf("Test complete.\n");
    }
    
    // Cleanup
    shmem_free(local_data);
    shmem_free(remote_data);
    
    shmem_finalize();
    
    return 0;
}
