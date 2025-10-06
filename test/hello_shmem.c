// OpenSHMEM function declarations
void shmem_init(void);
void shmem_finalize(void);
int shmem_my_pe(void);
int shmem_n_pes(void);
void *shmem_malloc(unsigned long size);
void shmem_free(void *ptr);
void shmem_barrier_all(void);
void shmem_put(void *dest, const void *source, unsigned long nelems, int pe);

// Printf for output
int printf(const char *format, ...);

int main(void) {
    // Initialize OpenSHMEM
    shmem_init();
    
    // Get my PE number and total number of PEs
    int me = shmem_my_pe();
    int npes = shmem_n_pes();
    
    // Allocate symmetric memory
    int *data = (int *)shmem_malloc(sizeof(int));
    *data = me;
    
    // Barrier to ensure all PEs have initialized
    shmem_barrier_all();
    
    // Simple put operation - PE 0 sends to PE 1
    if (me == 0 && npes > 1) {
        int value = 42;
        shmem_put(data, &value, 1, 1);
    }
    
    // Barrier to ensure put is complete
    shmem_barrier_all();
    
    // Print result
    printf("PE %d: data = %d\n", me, *data);
    
    // Free symmetric memory
    shmem_free(data);
    
    // Finalize
    shmem_finalize();
    
    return 0;
}
