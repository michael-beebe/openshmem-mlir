	.file	"LLVMDialectModule"
	.text
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	str	x30, [sp, #16]                  // 8-byte Folded Spill
	.cfi_def_cfa_offset 32
	.cfi_offset w30, -16
	bl	shmem_init
	bl	shmem_my_pe
	str	w0, [sp, #24]
	bl	shmem_n_pes
	ldr	w1, [sp, #24]
	mov	w2, w0
	str	w0, [sp, #12]
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	bl	printf
	bl	shmem_barrier_all
	bl	shmem_finalize
	ldr	x30, [sp, #16]                  // 8-byte Folded Reload
	mov	w0, wzr
	str	wzr, [sp, #28]
	add	sp, sp, #32
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.data
.L.str:
	.asciz	"Hello from PE %d of %d\n"
	.size	.L.str, 24

	.section	".note.GNU-stack","",@progbits
