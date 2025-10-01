// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @openshmem_test() {
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @openshmem_setup_test() {
func.func @openshmem_setup_test() -> () {
    // Setup operations from OpenSHMEMSetup.td
    
    // CHECK-NEXT: openshmem.init
    openshmem.init
    
    // CHECK-NEXT: [[vpe:%.*]] = openshmem.my_pe : i32
    %pe = openshmem.my_pe : i32
    
    // CHECK-NEXT: [[vnpes:%.*]] = openshmem.n_pes : i32
    %npes = openshmem.n_pes : i32
    
    // CHECK-NEXT: openshmem.finalize
    openshmem.finalize
    
    // CHECK-NEXT: return
    func.return
}

// Test OpenSHMEM region operation
// CHECK-LABEL: func.func @openshmem_region_test() {
func.func @openshmem_region_test() -> () {
    // CHECK-NEXT: openshmem.region {
    openshmem.region {
        // CHECK-NEXT: [[vpe:%.*]] = openshmem.my_pe : i32
        %pe = openshmem.my_pe : i32
        
        // CHECK-NEXT: [[vnpes:%.*]] = openshmem.n_pes : i32
        %npes = openshmem.n_pes : i32
    }
    
    // CHECK-NEXT: return
    func.return
}

// Test team operations from OpenSHMEMTeams.td
// CHECK-LABEL: func.func @openshmem_teams_test() {
func.func @openshmem_teams_test() -> () {
    openshmem.region {
        // Predefined teams
        // CHECK: [[vworld_team:%.*]] = openshmem.team_world -> !openshmem.team
        %world_team = openshmem.team_world -> !openshmem.team
        
        // CHECK: [[vshared_team:%.*]] = openshmem.team_shared -> !openshmem.team
        %shared_team = openshmem.team_shared -> !openshmem.team
        
        // Team queries
        // CHECK: [[vteam_pe:%.*]] = openshmem.team_my_pe([[vworld_team]]) : !openshmem.team -> i32
        %team_pe = openshmem.team_my_pe(%world_team) : !openshmem.team -> i32
        
        // CHECK: [[vteam_size:%.*]] = openshmem.team_n_pes([[vworld_team]]) : !openshmem.team -> i32
        %team_size = openshmem.team_n_pes(%world_team) : !openshmem.team -> i32
        
        // Team operations with constants for testing
        // CHECK: [[vstart:%.*]] = arith.constant 0 : i32
        %start = arith.constant 0 : i32
        // CHECK: [[vstride:%.*]] = arith.constant 2 : i32
        %stride = arith.constant 2 : i32
        // CHECK: [[vsize:%.*]] = arith.constant 4 : i32
        %size = arith.constant 4 : i32
        // CHECK: [[vxrange:%.*]] = arith.constant 2 : i32
        %xrange = arith.constant 2 : i32
        
        // Team split strided
        // CHECK: [[vnew_team:%.*]], [[vretval:%.*]] = openshmem.team_split_strided([[vworld_team]], [[vstart]], [[vstride]], [[vsize]]) : !openshmem.team, i32, i32, i32 -> !openshmem.team, i32
        %new_team, %retval = openshmem.team_split_strided(%world_team, %start, %stride, %size) : !openshmem.team, i32, i32, i32 -> !openshmem.team, i32
        
        // Team split 2D
        // CHECK: [[vxaxis_team:%.*]], [[vyaxis_team:%.*]], [[vretval2:%.*]] = openshmem.team_split_2d([[vworld_team]], [[vxrange]]) : !openshmem.team, i32 -> !openshmem.team, !openshmem.team, i32
        %xaxis_team, %yaxis_team, %retval2 = openshmem.team_split_2d(%world_team, %xrange) : !openshmem.team, i32 -> !openshmem.team, !openshmem.team, i32
        
        // Team synchronization
        // CHECK: openshmem.team_sync([[vnew_team]]) : !openshmem.team
        openshmem.team_sync(%new_team) : !openshmem.team
        
        // Clean up teams
        // CHECK: openshmem.team_destroy([[vnew_team]]) : !openshmem.team
        openshmem.team_destroy(%new_team) : !openshmem.team
        
        // CHECK: openshmem.team_destroy([[vxaxis_team]]) : !openshmem.team
        openshmem.team_destroy(%xaxis_team) : !openshmem.team
        
        // CHECK: openshmem.team_destroy([[vyaxis_team]]) : !openshmem.team
        openshmem.team_destroy(%yaxis_team) : !openshmem.team
    }
    
    // CHECK-NEXT: return
    func.return
}

// Test context operations from OpenSHMEMContexts.td
// CHECK-LABEL: func.func @openshmem_contexts_test() {
func.func @openshmem_contexts_test() -> () {
    openshmem.region {
        // Context creation constants
        // CHECK: [[voptions:%.*]] = arith.constant 0 : i64
        %options = arith.constant 0 : i64
        
        // Get world team for team context operations
        // CHECK: [[vworld_team:%.*]] = openshmem.team_world -> !openshmem.team
        %world_team = openshmem.team_world -> !openshmem.team
        
        // Create context
        // CHECK: [[vctx:%.*]], [[vstatus:%.*]] = openshmem.ctx_create([[voptions]]) : i64 -> !openshmem.ctx, i32
        %ctx, %status = openshmem.ctx_create(%options) : i64 -> !openshmem.ctx, i32
        
        // Create team context
        // CHECK: [[vteam_ctx:%.*]], [[vstatus2:%.*]] = openshmem.team_create_ctx([[vworld_team]], [[voptions]]) : !openshmem.team, i64 -> !openshmem.ctx, i32
        %team_ctx, %status2 = openshmem.team_create_ctx(%world_team, %options) : !openshmem.team, i64 -> !openshmem.ctx, i32
        
        // Get team from context
        // CHECK: [[vctx_team:%.*]], [[vstatus3:%.*]] = openshmem.ctx_get_team([[vctx]]) : !openshmem.ctx -> !openshmem.team, i32
        %ctx_team, %status3 = openshmem.ctx_get_team(%ctx) : !openshmem.ctx -> !openshmem.team, i32
        
        // Destroy contexts
        // CHECK: openshmem.ctx_destroy([[vctx]]) : !openshmem.ctx
        openshmem.ctx_destroy(%ctx) : !openshmem.ctx
        
        // CHECK: openshmem.ctx_destroy([[vteam_ctx]]) : !openshmem.ctx
        openshmem.ctx_destroy(%team_ctx) : !openshmem.ctx
    }
    
    // CHECK-NEXT: return
    func.return
}

// Test memory operations from OpenSHMEMMemory.td
// CHECK-LABEL: func.func @openshmem_memory_test() {
func.func @openshmem_memory_test() -> () {
    openshmem.region {
        // Memory allocation constants
        // CHECK: [[vsize:%.*]] = arith.constant 1024 : index
        %size = arith.constant 1024 : index
        // CHECK: [[vcount:%.*]] = arith.constant 256 : index
        %count = arith.constant 256 : index
        // CHECK: [[valignment:%.*]] = arith.constant 64 : index
        %alignment = arith.constant 64 : index
        // CHECK: [[voffset:%.*]] = arith.constant 10 : index
        %offset = arith.constant 10 : index
        
        // Basic memory allocation
        // CHECK: [[vmem:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi32, #openshmem.symmetric_memory>
        %mem = openshmem.malloc(%size) : index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Zero-initialized allocation
        // CHECK: [[vmem2:%.*]] = openshmem.calloc([[vcount]], [[vsize]]) : index, index -> memref<?xi32, #openshmem.symmetric_memory>
        %mem2 = openshmem.calloc(%count, %size) : index, index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Aligned allocation
        // CHECK: [[vmem3:%.*]] = openshmem.align([[valignment]], [[vsize]]) : index, index -> memref<?xi32, #openshmem.symmetric_memory>
        %mem3 = openshmem.align(%alignment, %size) : index, index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Memory reallocation
        // CHECK: [[vmem4:%.*]] = openshmem.realloc([[vmem]], [[vsize]]) : memref<?xi32, #openshmem.symmetric_memory>, index -> memref<?xi32, #openshmem.symmetric_memory>
        %mem4 = openshmem.realloc(%mem, %size) : memref<?xi32, #openshmem.symmetric_memory>, index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Memory offset calculation
        // CHECK: [[vmem_offset:%.*]] = openshmem.offset([[vmem2]], [[voffset]]) : memref<?xi32, #openshmem.symmetric_memory>, index -> memref<?xi32, #openshmem.symmetric_memory>
        %mem_offset = openshmem.offset(%mem2, %offset) : memref<?xi32, #openshmem.symmetric_memory>, index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Free memory (in reverse order of dependencies)
        // CHECK: openshmem.free([[vmem4]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%mem4) : memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: openshmem.free([[vmem3]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%mem3) : memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: openshmem.free([[vmem2]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%mem2) : memref<?xi32, #openshmem.symmetric_memory>
    }
    
    // CHECK-NEXT: return
    func.return
}

// Test synchronization operations from OpenSHMEMSync.td
// CHECK-LABEL: func.func @openshmem_sync_test() {
func.func @openshmem_sync_test() -> () {
    openshmem.region {
        // Global barrier
        // CHECK: openshmem.barrier_all
        openshmem.barrier_all
        
        // Quiet operation (ensure RMA completion)
        // CHECK: openshmem.quiet
        openshmem.quiet
        
        // Legacy barrier with parameters (deprecated but still supported)
        // Create sync array for legacy barrier
        // CHECK: [[vsize:%.*]] = arith.constant 64 : index
        %size = arith.constant 64 : index
        // CHECK: [[vpsync:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi64, #openshmem.symmetric_memory>
        %psync = openshmem.malloc(%size) : index -> memref<?xi64, #openshmem.symmetric_memory>
        
        // Legacy barrier parameters
        // CHECK: [[vpe_start:%.*]] = arith.constant 0 : i32
        %pe_start = arith.constant 0 : i32
        // CHECK: [[vlog_pe_stride:%.*]] = arith.constant 0 : i32
        %log_pe_stride = arith.constant 0 : i32
        // CHECK: [[vpe_size:%.*]] = arith.constant 4 : i32
        %pe_size = arith.constant 4 : i32
        
        // Legacy barrier operation
        // CHECK: openshmem.barrier([[vpe_start]], [[vlog_pe_stride]], [[vpe_size]], [[vpsync]]) : i32, i32, i32, memref<?xi64, #openshmem.symmetric_memory>
        openshmem.barrier(%pe_start, %log_pe_stride, %pe_size, %psync) : i32, i32, i32, memref<?xi64, #openshmem.symmetric_memory>
        
        // Clean up
        // CHECK: openshmem.free([[vpsync]]) : memref<?xi64, #openshmem.symmetric_memory>
        openshmem.free(%psync) : memref<?xi64, #openshmem.symmetric_memory>
    }
    
    // CHECK-NEXT: return
    func.return
}

// Test point-to-point synchronization operations from OpenSHMEMPt2ptSync.td
// CHECK-LABEL: func.func @openshmem_pt2pt_sync_test() {
func.func @openshmem_pt2pt_sync_test() -> () {
    openshmem.region {
        // Allocate test variables
        // CHECK: [[vsize:%.*]] = arith.constant 4 : index
        %size = arith.constant 4 : index
        // CHECK: [[varray_size:%.*]] = arith.constant 10 : index
        %array_size = arith.constant 10 : index
        
        // Single variable for simple wait/test
        // CHECK: [[vivar:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi32, #openshmem.symmetric_memory>
        %ivar = openshmem.malloc(%size) : index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Array of variables for bulk operations
        // CHECK: [[vivars:%.*]] = openshmem.malloc([[varray_size]]) : index -> memref<?xi32, #openshmem.symmetric_memory>
        %ivars = openshmem.malloc(%array_size) : index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Status and indices arrays (regular memory, not symmetric)
        // CHECK: [[vstatus:%.*]] = memref.alloc() : memref<10xi32>
        %status = memref.alloc() : memref<10xi32>
        // CHECK: [[vindices:%.*]] = memref.alloc() : memref<10xindex>
        %indices = memref.alloc() : memref<10xindex>
        
        // Signal address for signal operations
        // CHECK: [[vsig_addr:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi64, #openshmem.symmetric_memory>
        %sig_addr = openshmem.malloc(%size) : index -> memref<?xi64, #openshmem.symmetric_memory>
        
        // Test constants
        // CHECK: [[vcmp:%.*]] = arith.constant 1 : i32
        %cmp = arith.constant 1 : i32  // SHMEM_CMP_EQ equivalent
        // CHECK: [[vcmp_value:%.*]] = arith.constant 42 : i32
        %cmp_value = arith.constant 42 : i32
        // CHECK: [[vcmp_value_64:%.*]] = arith.constant 100 : i64
        %cmp_value_64 = arith.constant 100 : i64
        // CHECK: [[vnelems:%.*]] = arith.constant 5 : index
        %nelems = arith.constant 5 : index
        
        // Basic wait until operation
        // CHECK: openshmem.wait_until([[vivar]], [[vcmp]], [[vcmp_value]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        openshmem.wait_until(%ivar, %cmp, %cmp_value) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        
        // Basic test operation
        // CHECK: [[vtest_result:%.*]] = openshmem.test([[vivar]], [[vcmp]], [[vcmp_value]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        %test_result = openshmem.test(%ivar, %cmp, %cmp_value) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        
        // Wait until all variables satisfy condition
        // CHECK: openshmem.wait_until_all([[vivars]], [[vnelems]], [[vstatus]], [[vcmp]], [[vcmp_value]]) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xi32>, i32, i32
        openshmem.wait_until_all(%ivars, %nelems, %status, %cmp, %cmp_value) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xi32>, i32, i32
        
        // Wait until any variable satisfies condition
        // CHECK: openshmem.wait_until_any([[vivars]], [[vnelems]], [[vstatus]], [[vcmp]], [[vcmp_value]]) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xi32>, i32, i32
        openshmem.wait_until_any(%ivars, %nelems, %status, %cmp, %cmp_value) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xi32>, i32, i32
        
        // Wait until some variables satisfy condition
        // CHECK: [[vsome_result:%.*]] = openshmem.wait_until_some([[vivars]], [[vnelems]], [[vindices]], [[vstatus]], [[vcmp]], [[vcmp_value]]) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xindex>, memref<10xi32>, i32, i32 -> index
        %some_result = openshmem.wait_until_some(%ivars, %nelems, %indices, %status, %cmp, %cmp_value) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xindex>, memref<10xi32>, i32, i32 -> index
        
        // Test all variables
        // CHECK: [[vtest_all_result:%.*]] = openshmem.test_all([[vivars]], [[vnelems]], [[vstatus]], [[vcmp]], [[vcmp_value]]) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xi32>, i32, i32 -> i32
        %test_all_result = openshmem.test_all(%ivars, %nelems, %status, %cmp, %cmp_value) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xi32>, i32, i32 -> i32
        
        // Test any variable
        // CHECK: [[vtest_any_result:%.*]] = openshmem.test_any([[vivars]], [[vnelems]], [[vstatus]], [[vcmp]], [[vcmp_value]]) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xi32>, i32, i32 -> index
        %test_any_result = openshmem.test_any(%ivars, %nelems, %status, %cmp, %cmp_value) : memref<?xi32, #openshmem.symmetric_memory>, index, memref<10xi32>, i32, i32 -> index
        
        // Signal wait until operation
        // CHECK: [[vsignal_result:%.*]] = openshmem.signal_wait_until([[vsig_addr]], [[vcmp]], [[vcmp_value_64]]) : memref<?xi64, #openshmem.symmetric_memory>, i32, i64 -> i64
        %signal_result = openshmem.signal_wait_until(%sig_addr, %cmp, %cmp_value_64) : memref<?xi64, #openshmem.symmetric_memory>, i32, i64 -> i64
        
        // Clean up symmetric memory
        // CHECK: openshmem.free([[vivar]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%ivar) : memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: openshmem.free([[vivars]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%ivars) : memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: openshmem.free([[vsig_addr]]) : memref<?xi64, #openshmem.symmetric_memory>
        openshmem.free(%sig_addr) : memref<?xi64, #openshmem.symmetric_memory>
        
        // Clean up regular memory
        // CHECK: memref.dealloc [[vstatus]] : memref<10xi32>
        memref.dealloc %status : memref<10xi32>
        
        // CHECK: memref.dealloc [[vindices]] : memref<10xindex>
        memref.dealloc %indices : memref<10xindex>
    }
    
    // CHECK-NEXT: return
    func.return
}

// Test atomic operations from OpenSHMEMAtomics.td
// CHECK-LABEL: func.func @openshmem_atomics_test() {
func.func @openshmem_atomics_test() -> () {
    openshmem.region {
        // Allocate memory for atomic operations
        // CHECK: [[vsize:%.*]] = arith.constant 4 : index
        %size = arith.constant 4 : index
        
        // Symmetric memory for atomic targets
        // CHECK: [[vdest:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi32, #openshmem.symmetric_memory>
        %dest = openshmem.malloc(%size) : index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: [[vsource:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi32, #openshmem.symmetric_memory>
        %source = openshmem.malloc(%size) : index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Create a context for context-aware operations
        // CHECK: [[voptions:%.*]] = arith.constant 0 : i64
        %options = arith.constant 0 : i64
        // CHECK: [[vctx:%.*]], [[vstatus:%.*]] = openshmem.ctx_create([[voptions]]) : i64 -> !openshmem.ctx, i32
        %ctx, %status = openshmem.ctx_create(%options) : i64 -> !openshmem.ctx, i32
        
        // Test constants
        // CHECK: [[vpe:%.*]] = arith.constant 1 : i32
        %pe = arith.constant 1 : i32
        // CHECK: [[vvalue:%.*]] = arith.constant 42 : i32
        %value = arith.constant 42 : i32
        // CHECK: [[vcond:%.*]] = arith.constant 10 : i32
        %cond = arith.constant 10 : i32
        // CHECK: [[vnew_value:%.*]] = arith.constant 99 : i32
        %new_value = arith.constant 99 : i32
        
        // Basic atomic fetch
        // CHECK: [[vfetch_result:%.*]] = openshmem.atomic_fetch([[vsource]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32 -> i32
        %fetch_result = openshmem.atomic_fetch(%source, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32 -> i32
        
        // Context-aware atomic fetch
        // CHECK: [[vctx_fetch_result:%.*]] = openshmem.ctx_atomic_fetch([[vctx]], [[vsource]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32 -> i32
        %ctx_fetch_result = openshmem.ctx_atomic_fetch(%ctx, %source, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32 -> i32
        
        // Atomic set
        // CHECK: openshmem.atomic_set([[vdest]], [[vvalue]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        openshmem.atomic_set(%dest, %value, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        
        // Context-aware atomic set
        // CHECK: openshmem.ctx_atomic_set([[vctx]], [[vdest]], [[vvalue]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        openshmem.ctx_atomic_set(%ctx, %dest, %value, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        
        // Atomic swap
        // CHECK: [[vswap_result:%.*]] = openshmem.atomic_swap([[vdest]], [[vvalue]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        %swap_result = openshmem.atomic_swap(%dest, %value, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        
        // Context-aware atomic swap
        // CHECK: [[vctx_swap_result:%.*]] = openshmem.ctx_atomic_swap([[vctx]], [[vdest]], [[vvalue]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        %ctx_swap_result = openshmem.ctx_atomic_swap(%ctx, %dest, %value, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        
        // Atomic compare-and-swap
        // CHECK: [[vcas_result:%.*]] = openshmem.atomic_compare_swap([[vdest]], [[vcond]], [[vnew_value]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32, i32 -> i32
        %cas_result = openshmem.atomic_compare_swap(%dest, %cond, %new_value, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32, i32 -> i32
        
        // Context-aware atomic compare-and-swap
        // CHECK: [[vctx_cas_result:%.*]] = openshmem.ctx_atomic_compare_swap([[vctx]], [[vdest]], [[vcond]], [[vnew_value]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32, i32 -> i32
        %ctx_cas_result = openshmem.ctx_atomic_compare_swap(%ctx, %dest, %cond, %new_value, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32, i32 -> i32
        
        // Atomic fetch-and-add
        // CHECK: [[vfetch_add_result:%.*]] = openshmem.atomic_fetch_add([[vdest]], [[vvalue]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        %fetch_add_result = openshmem.atomic_fetch_add(%dest, %value, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        
        // Context-aware atomic fetch-and-add
        // CHECK: [[vctx_fetch_add_result:%.*]] = openshmem.ctx_atomic_fetch_add([[vctx]], [[vdest]], [[vvalue]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        %ctx_fetch_add_result = openshmem.ctx_atomic_fetch_add(%ctx, %dest, %value, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        
        // Atomic add (no return value)
        // CHECK: openshmem.atomic_add([[vdest]], [[vvalue]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        openshmem.atomic_add(%dest, %value, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        
        // Context-aware atomic add
        // CHECK: openshmem.ctx_atomic_add([[vctx]], [[vdest]], [[vvalue]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        openshmem.ctx_atomic_add(%ctx, %dest, %value, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32, i32
        
        // Atomic fetch-and-increment
        // CHECK: [[vfetch_inc_result:%.*]] = openshmem.atomic_fetch_inc([[vdest]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32 -> i32
        %fetch_inc_result = openshmem.atomic_fetch_inc(%dest, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32 -> i32
        
        // Context-aware atomic fetch-and-increment
        // CHECK: [[vctx_fetch_inc_result:%.*]] = openshmem.ctx_atomic_fetch_inc([[vctx]], [[vdest]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32 -> i32
        %ctx_fetch_inc_result = openshmem.ctx_atomic_fetch_inc(%ctx, %dest, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32 -> i32
        
        // Atomic increment (no return value)
        // CHECK: openshmem.atomic_inc([[vdest]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32
        openshmem.atomic_inc(%dest, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32
        
        // Context-aware atomic increment
        // CHECK: openshmem.ctx_atomic_inc([[vctx]], [[vdest]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32
        openshmem.ctx_atomic_inc(%ctx, %dest, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, i32
        
        // Bitwise atomic operations
        // CHECK: [[vfetch_and_result:%.*]] = openshmem.atomic_fetch_and([[vdest]], [[vvalue]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        %fetch_and_result = openshmem.atomic_fetch_and(%dest, %value, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        
        // CHECK: [[vfetch_or_result:%.*]] = openshmem.atomic_fetch_or([[vdest]], [[vvalue]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        %fetch_or_result = openshmem.atomic_fetch_or(%dest, %value, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        
        // CHECK: [[vfetch_xor_result:%.*]] = openshmem.atomic_fetch_xor([[vdest]], [[vvalue]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        %fetch_xor_result = openshmem.atomic_fetch_xor(%dest, %value, %pe) : memref<?xi32, #openshmem.symmetric_memory>, i32, i32 -> i32
        
        // Clean up
        // CHECK: openshmem.ctx_destroy([[vctx]]) : !openshmem.ctx
        openshmem.ctx_destroy(%ctx) : !openshmem.ctx
        
        // CHECK: openshmem.free([[vdest]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%dest) : memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: openshmem.free([[vsource]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%source) : memref<?xi32, #openshmem.symmetric_memory>
    }
    
    // CHECK-NEXT: return
    func.return
}

// Test collective operations from OpenSHMEMCollectives.td
// CHECK-LABEL: func.func @openshmem_collectives_test() {
func.func @openshmem_collectives_test() -> () {
    openshmem.region {
        // Setup memory for collective operations
        // CHECK: [[vsize:%.*]] = arith.constant 100 : index
        %size = arith.constant 100 : index
        // CHECK: [[vnelems:%.*]] = arith.constant 25 : index
        %nelems = arith.constant 25 : index
        
        // Symmetric memory for collective operations
        // CHECK: [[vsource:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi32, #openshmem.symmetric_memory>
        %source = openshmem.malloc(%size) : index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: [[vdest:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi32, #openshmem.symmetric_memory>
        %dest = openshmem.malloc(%size) : index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Setup team for collective operations
        // CHECK: [[vteam:%.*]] = openshmem.team_world -> !openshmem.team
        %team = openshmem.team_world -> !openshmem.team
        
        // Constants for collective operations
        // CHECK: [[vroot_pe:%.*]] = arith.constant 0 : i32
        %root_pe = arith.constant 0 : i32
        // CHECK: [[vstride:%.*]] = arith.constant 1 : index
        %stride = arith.constant 1 : index
        // CHECK: [[vdst_stride:%.*]] = arith.constant 1 : index
        %dst_stride = arith.constant 1 : index
        // CHECK: [[vsrc_stride:%.*]] = arith.constant 1 : index
        %src_stride = arith.constant 1 : index
        
        // ALL-TO-ALL operations
        
        // Basic all-to-all (typed)
        // CHECK: [[valltoall_ret:%.*]] = openshmem.alltoall([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %alltoall_ret = openshmem.alltoall(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // All-to-all memory (raw bytes)
        // CHECK: [[valltoallmem_ret:%.*]] = openshmem.alltoallmem([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %alltoallmem_ret = openshmem.alltoallmem(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // All-to-all strided (typed)
        // CHECK: [[valltoalls_ret:%.*]] = openshmem.alltoalls([[vteam]], [[vdest]], [[vsource]], [[vdst_stride]], [[vsrc_stride]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, index, index -> i32
        %alltoalls_ret = openshmem.alltoalls(%team, %dest, %source, %dst_stride, %src_stride, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, index, index -> i32
        
        // All-to-all strided memory (raw bytes)
        // CHECK: [[valltoallsmem_ret:%.*]] = openshmem.alltoallsmem([[vteam]], [[vdest]], [[vsource]], [[vdst_stride]], [[vsrc_stride]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, index, index -> i32
        %alltoallsmem_ret = openshmem.alltoallsmem(%team, %dest, %source, %dst_stride, %src_stride, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, index, index -> i32
        
        // BROADCAST operations
        
        // Broadcast (typed)
        // CHECK: [[vbroadcast_ret:%.*]] = openshmem.broadcast([[vteam]], [[vdest]], [[vsource]], [[vnelems]], [[vroot_pe]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, i32 -> i32
        %broadcast_ret = openshmem.broadcast(%team, %dest, %source, %nelems, %root_pe) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, i32 -> i32
        
        // Broadcast memory (raw bytes)
        // CHECK: [[vbroadcastmem_ret:%.*]] = openshmem.broadcastmem([[vteam]], [[vdest]], [[vsource]], [[vnelems]], [[vroot_pe]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, i32 -> i32
        %broadcastmem_ret = openshmem.broadcastmem(%team, %dest, %source, %nelems, %root_pe) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, i32 -> i32
        
        // COLLECT operations (gather to all)
        
        // Collect (typed)
        // CHECK: [[vcollect_ret:%.*]] = openshmem.collect([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %collect_ret = openshmem.collect(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Collect memory (raw bytes)
        // CHECK: [[vcollectmem_ret:%.*]] = openshmem.collectmem([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %collectmem_ret = openshmem.collectmem(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Fixed-size collect (fcollect) - all PEs contribute the same amount
        // CHECK: [[vfcollect_ret:%.*]] = openshmem.fcollect([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %fcollect_ret = openshmem.fcollect(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Fixed-size collect memory (raw bytes)
        // CHECK: [[vfcollectmem_ret:%.*]] = openshmem.fcollectmem([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %fcollectmem_ret = openshmem.fcollectmem(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // REDUCTION operations
        
        // Sum reduction
        // CHECK: [[vsumreduce_ret:%.*]] = openshmem.sumreduce([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %sumreduce_ret = openshmem.sumreduce(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Product reduction
        // CHECK: [[vprodreduce_ret:%.*]] = openshmem.prodreduce([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %prodreduce_ret = openshmem.prodreduce(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Min reduction
        // CHECK: [[vminreduce_ret:%.*]] = openshmem.minreduce([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %minreduce_ret = openshmem.minreduce(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Max reduction
        // CHECK: [[vmaxreduce_ret:%.*]] = openshmem.maxreduce([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %maxreduce_ret = openshmem.maxreduce(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Bitwise AND reduction
        // CHECK: [[vandreduce_ret:%.*]] = openshmem.andreduce([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %andreduce_ret = openshmem.andreduce(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Bitwise OR reduction
        // CHECK: [[vorreduce_ret:%.*]] = openshmem.orreduce([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %orreduce_ret = openshmem.orreduce(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Bitwise XOR reduction
        // CHECK: [[vxorreduce_ret:%.*]] = openshmem.xorreduce([[vteam]], [[vdest]], [[vsource]], [[vnelems]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        %xorreduce_ret = openshmem.xorreduce(%team, %dest, %source, %nelems) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
        
        // Clean up
        // CHECK: openshmem.free([[vsource]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%source) : memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: openshmem.free([[vdest]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%dest) : memref<?xi32, #openshmem.symmetric_memory>
    }
    
    // CHECK-NEXT: return
    func.return
}

// Test RMA operations from OpenSHMEMRMAOps.td
// CHECK-LABEL: func.func @openshmem_rma_test() {
func.func @openshmem_rma_test() -> () {
    openshmem.region {
        // Allocate memory for RMA operations
        // CHECK: [[vsize:%.*]] = arith.constant 100 : index
        %size = arith.constant 100 : index
        // CHECK: [[vnelems:%.*]] = arith.constant 25 : index
        %nelems = arith.constant 25 : index
        // CHECK: [[vbytes:%.*]] = arith.constant 400 : index
        %bytes = arith.constant 400 : index
        
        // Local memory (source for puts, destination for gets)
        // CHECK: [[vlocal_mem:%.*]] = memref.alloc() : memref<100xi32>
        %local_mem = memref.alloc() : memref<100xi32>
        
        // Symmetric memory (destination for puts, source for gets)
        // CHECK: [[vsymm_mem:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi32, #openshmem.symmetric_memory>
        %symm_mem = openshmem.malloc(%size) : index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Another symmetric memory for testing
        // CHECK: [[vsymm_mem2:%.*]] = openshmem.malloc([[vsize]]) : index -> memref<?xi32, #openshmem.symmetric_memory>
        %symm_mem2 = openshmem.malloc(%size) : index -> memref<?xi32, #openshmem.symmetric_memory>
        
        // Create a context for context-aware operations
        // CHECK: [[voptions:%.*]] = arith.constant 0 : i64
        %options = arith.constant 0 : i64
        // CHECK: [[vctx:%.*]], [[vstatus:%.*]] = openshmem.ctx_create([[voptions]]) : i64 -> !openshmem.ctx, i32
        %ctx, %status = openshmem.ctx_create(%options) : i64 -> !openshmem.ctx, i32
        
        // Target PE
        // CHECK: [[vpe:%.*]] = arith.constant 1 : i32
        %pe = arith.constant 1 : i32
        
        // PUT operations (local -> remote symmetric)
        
        // Basic blocking put
        // CHECK: openshmem.put([[vsymm_mem]], [[vlocal_mem]], [[vnelems]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        openshmem.put(%symm_mem, %local_mem, %nelems, %pe) : memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        
        // Context-aware blocking put
        // CHECK: openshmem.ctx_put([[vctx]], [[vsymm_mem]], [[vlocal_mem]], [[vnelems]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        openshmem.ctx_put(%ctx, %symm_mem, %local_mem, %nelems, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        
        // Non-blocking put
        // CHECK: openshmem.put_nbi([[vsymm_mem]], [[vlocal_mem]], [[vnelems]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        openshmem.put_nbi(%symm_mem, %local_mem, %nelems, %pe) : memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        
        // Context-aware non-blocking put
        // CHECK: openshmem.ctx_put_nbi([[vctx]], [[vsymm_mem]], [[vlocal_mem]], [[vnelems]], [[vpe]]) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        openshmem.ctx_put_nbi(%ctx, %symm_mem, %local_mem, %nelems, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        
        // Memory-based put (putmem) - raw bytes
        // CHECK: openshmem.putmem([[vsymm_mem]], [[vlocal_mem]], [[vbytes]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        openshmem.putmem(%symm_mem, %local_mem, %bytes, %pe) : memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        
        // Non-blocking memory-based put
        // CHECK: openshmem.putmem_nbi([[vsymm_mem]], [[vlocal_mem]], [[vbytes]], [[vpe]]) : memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        openshmem.putmem_nbi(%symm_mem, %local_mem, %bytes, %pe) : memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
        
        // GET operations (remote symmetric -> local)
        
        // Basic blocking get
        // CHECK: openshmem.get([[vlocal_mem]], [[vsymm_mem2]], [[vnelems]], [[vpe]]) : memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        openshmem.get(%local_mem, %symm_mem2, %nelems, %pe) : memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        
        // Context-aware blocking get
        // CHECK: openshmem.ctx_get([[vctx]], [[vlocal_mem]], [[vsymm_mem2]], [[vnelems]], [[vpe]]) : !openshmem.ctx, memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        openshmem.ctx_get(%ctx, %local_mem, %symm_mem2, %nelems, %pe) : !openshmem.ctx, memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        
        // Non-blocking get
        // CHECK: openshmem.get_nbi([[vlocal_mem]], [[vsymm_mem2]], [[vnelems]], [[vpe]]) : memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        openshmem.get_nbi(%local_mem, %symm_mem2, %nelems, %pe) : memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        
        // Context-aware non-blocking get
        // CHECK: openshmem.ctx_get_nbi([[vctx]], [[vlocal_mem]], [[vsymm_mem2]], [[vnelems]], [[vpe]]) : !openshmem.ctx, memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        openshmem.ctx_get_nbi(%ctx, %local_mem, %symm_mem2, %nelems, %pe) : !openshmem.ctx, memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        
        // Memory-based get (getmem) - raw bytes
        // CHECK: openshmem.getmem([[vlocal_mem]], [[vsymm_mem2]], [[vbytes]], [[vpe]]) : memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        openshmem.getmem(%local_mem, %symm_mem2, %bytes, %pe) : memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        
        // Non-blocking memory-based get
        // CHECK: openshmem.getmem_nbi([[vlocal_mem]], [[vsymm_mem2]], [[vbytes]], [[vpe]]) : memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        openshmem.getmem_nbi(%local_mem, %symm_mem2, %bytes, %pe) : memref<100xi32>, memref<?xi32, #openshmem.symmetric_memory>, index, i32
        
        // Completion operations to ensure non-blocking operations finish
        // CHECK: openshmem.quiet
        openshmem.quiet
        
        // Clean up
        // CHECK: openshmem.ctx_destroy([[vctx]]) : !openshmem.ctx
        openshmem.ctx_destroy(%ctx) : !openshmem.ctx
        
        // CHECK: openshmem.free([[vsymm_mem]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%symm_mem) : memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: openshmem.free([[vsymm_mem2]]) : memref<?xi32, #openshmem.symmetric_memory>
        openshmem.free(%symm_mem2) : memref<?xi32, #openshmem.symmetric_memory>
        
        // CHECK: memref.dealloc [[vlocal_mem]] : memref<100xi32>
        memref.dealloc %local_mem : memref<100xi32>
    }
    
    // CHECK-NEXT: return
    func.return
}
