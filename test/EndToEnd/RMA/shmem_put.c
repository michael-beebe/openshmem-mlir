#include <stdio.h>
#include <shmem.h>

int main() {
  shmem_init();

  int me = shmem_my_pe();
  int npes = shmem_n_pes();

  static int src = 42;
  static int dest = 0;

  shmem_barrier_all();

  if (me == 0) {
    shmem_put(&dest, &src, 1, 1); // PE 0 puts src to dest on PE 1
  }

  shmem_barrier_all();

  if (me == 1) {
    printf("PE %d: dest = %d\n", me, dest);
  }

  shmem_finalize();
  return 0;
}
