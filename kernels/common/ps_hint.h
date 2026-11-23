// ps_hint.h -- PhaseShift device-side memory-class hints and work-unit markers (ps patch mem-class).
//
// Two custom user read/write CSRs of the ps tree (vortex-trees/ps/VX_types.toml [csr_ps]; RISC-V custom range
// 0x800-0x8FF, errata 16):
//   VX_CSR_PS_MEM_HINT 0x8C0  per-warp class of the warp's later loads/stores (PS_CLASS_*; 0 = unset, the address
//                             ranges of [dcr_phase] decide); SimX tags the LSU requests with it (methodology §12)
//   VX_CSR_PS_MARK     0x8C1  a write counts one work unit of type value & 7 (PS_MARK_*) on the writing core PER ACTIVE
//                             SFU-PACKET TRACE (SimX splits a warp instruction into one trace per SFU packet with an
//                             active lane when NUM_SFU_LANES < NUM_THREADS; errata 33b): write it from ONE lane only
//                             (`if (tid == 0)`); read = the core's total in this launch
//
// The macros compile to NOTHING unless the kernel is built with -DPS_HINTS=1 (passed through CONFIGS, e.g. vxrun
// `configs: -DPS_HINTS=1`), so a PS_HINTS=0 build of an validated kernel is byte-identical to its build without this
// header (checked: results/<run>/elf_sha_*.txt). A PS_HINTS=1 binary aborts on the base tree (unknown CSR,
// sim/simx/csr_unit.cpp) by design, and it is a DIFFERENT binary: compare mechanisms only between configurations
// running the same binary (methodology §12.7).
//
// A hint is read by the LSU when it builds a request, and csrw does not stall the warp: it takes effect reliably only
// across a CTA barrier or a data dependency (methodology §12.3).
// Usage in a kernel (every warp executes PS_MEM_HINT for itself, because the hint is per warp):
//   PS_MEM_HINT(PS_CLASS_KV_STREAM);  ... K/V loads ...  PS_MEM_HINT(PS_CLASS_ORDINARY);
//   if (threadIdx.x == 0) PS_MARK(PS_MARK_DECODE_TILE);   // one warp per CTA marks each finished tile
// Host code may include this header too (for the constants); the CSR macros are device-only.
#ifndef PS_HINT_H
#define PS_HINT_H

// Constants spelled out so this header compiles against the base tree's VX_types.h as well; checked against the
// generated header when it has them (ps tree).
#define PS_CSR_MEM_HINT 0x8C0
#define PS_CSR_MARK 0x8C1

#define PS_CLASS_ORDINARY 0u
#define PS_CLASS_WEIGHT 1u
#define PS_CLASS_KV_PERSIST 2u
#define PS_CLASS_KV_STREAM 3u
#define PS_CLASS_BULK_DXA 4u

#define PS_MARK_OTHER 0u
#define PS_MARK_PREFILL_TILE 1u
#define PS_MARK_DECODE_TILE 2u

#ifdef VX_CSR_PS_MEM_HINT
#if (VX_CSR_PS_MEM_HINT != PS_CSR_MEM_HINT) || (VX_CSR_PS_MARK != PS_CSR_MARK) ||                              \
    (VX_PS_CLASS_ORDINARY != 0) || (VX_PS_CLASS_WEIGHT != 1) || (VX_PS_CLASS_KV_PERSIST != 2) ||               \
    (VX_PS_CLASS_KV_STREAM != 3) || (VX_PS_CLASS_BULK_DXA != 4) || (VX_PS_MARK_OTHER != 0) ||                  \
    (VX_PS_MARK_PREFILL_TILE != 1) || (VX_PS_MARK_DECODE_TILE != 2)
#error "ps_hint.h constants disagree with VX_types.h ([csr_ps], [ps_mem_class], [ps_mark_type])"
#endif
#endif

#ifndef PS_HINTS
#define PS_HINTS 0
#endif

#if PS_HINTS && defined(__VORTEX__)
// csrw <csr>, <reg>: a CSRRW with rd = x0 (no read-back needed). volatile + memory clobber: the hint must not move
// across the loads/stores it classifies, and a mark must not be merged or dropped.
#define PS_STR_(x) #x
#define PS_STR(x) PS_STR_(x)
#define PS_MEM_HINT(cls) __asm__ volatile("csrw " PS_STR(PS_CSR_MEM_HINT) ", %0" ::"r"((unsigned)(cls)) : "memory")
#define PS_MARK(type) __asm__ volatile("csrw " PS_STR(PS_CSR_MARK) ", %0" ::"r"((unsigned)(type)) : "memory")
#else
#define PS_MEM_HINT(cls) ((void)0)
#define PS_MARK(type) ((void)0)
#endif

#endif  // PS_HINT_H
