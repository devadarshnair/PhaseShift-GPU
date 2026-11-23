// kernel.cpp -- ps_issueprobe device kernels (/.md #1(a)).
//
// Purpose: measure the per-warp issue cadence tau (cycles between two consecutive instructions of ONE warp) of the
// Vortex 3.0 core, per instruction class, as a function of the number of resident warps W. Model (explained): with W
// warps ready, core IPC = W / tau as long as tau is a per-warp serial limit and the core front end (1 instruction per
// cycle for the whole core, SimX schedule()) is not saturated; IPC(W) saturates at the shared limit. Fitting
// IPC(W) = W/tau on W = 1, 2, 4 shows the per-warp limit, IPC(8) shows the shared limit.
//
// Every kernel is a loop of `iters` iterations; one iteration is U = 32 inline-asm instructions (written as separate
// `asm volatile` statements, so the compiler can neither reorder nor drop them) plus the compiler's loop control
// (counter decrement + one backward branch; verified in the kernel disassembly, see results/<run>/README.md).
//   alu_ind  : 16 independent chains of `add a_k, a_k, b`
//   alu_dep  : one chain of `add a, a, b`
//   fma_ind4/8/16 : 4/8/16 independent chains of `fmadd.s a_k, a_k, b, c`   (the existing ps_memprobe fma uses 8)
//   fma_dep  : one chain of fmadd.s
//   fadd_ind : 8 independent chains of `fadd.s a_k, a_k, b`
//   fsgnj_ind: 8 independent chains of `fsgnj.s a_k, a_k, b`   (FPU-unit op that does NOT write FFLAGS: control)
//   fcvt_ind : 8 independent destinations of `fcvt.s.w f_k, rb`  (int->float, writes FFLAGS)
//   mix_fa   : 8 fmadd.s chains interleaved 1:1 with 8 add chains
//   ld_ind   : 32 independent `lw` (16 different destination registers; 8 distinct 256 B-spaced lines), L1-resident,
//              one coalesced 128 B access per warp-load
//   ld_dep   : one dependent `lw p, 0(p)` chain (self-pointing cell, L1 hit)
//   br8/br4  : taken `j .+4` (JAL, parks the warp until commit) after every 7 / 3 add instructions
// Results are host-checked (bit-exact), see main.cpp. The kernels exist to time the pipeline, not to compute.
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#define ADD(a)   __asm__ volatile("add %0, %0, %1" : "+r"(a) : "r"(b))
#define FMA(a)   __asm__ volatile("fmadd.s %0, %0, %1, %2" : "+f"(a) : "f"(fb), "f"(fc))
#define FADD(a)  __asm__ volatile("fadd.s %0, %0, %1" : "+f"(a) : "f"(fb))
#define FSGN(a)  __asm__ volatile("fsgnj.s %0, %0, %1" : "+f"(a) : "f"(fb))
#define FCVT(a)  __asm__ volatile("fcvt.s.w %0, %1" : "=f"(a) : "r"(b))
#define JMP()    __asm__ volatile("j 1f\n1:")
#define LD(r, k) __asm__ volatile("lw %0, %2(%1)" : "=&r"(r) : "r"(p), "I"((k) * 256))

struct Ctx {
  uint32_t gid, warp, lane, n;
  bool lead;
};
static inline Ctx ctx(const ps_issueprobe_arg_t* arg) {
  Ctx c;
  c.gid = blockIdx.x * blockDim.x + threadIdx.x;
  c.warp = threadIdx.x >> 5;       // NT = 32 (PS-BASE-S0)
  c.lane = threadIdx.x & 31u;
  c.n = arg->iters;
  c.lead = (c.lane == 0);
  return c;
}
static inline void emit(const ps_issueprobe_arg_t* arg, const Ctx& c, uint32_t t0, uint32_t t1, uint32_t res) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out_addr);
  uint32_t* cyc = reinterpret_cast<uint32_t*>(arg->cyc_addr);
  out[2 * c.gid] = res;
  if (c.lead) { cyc[2 * c.warp] = t0; cyc[2 * c.warp + 1] = t1; }
}
static inline void emitf(const ps_issueprobe_arg_t* arg, const Ctx& c, uint32_t t0, uint32_t t1, float res, uint32_t ires) {
  float* outf = reinterpret_cast<float*>(arg->out_addr);
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out_addr);
  outf[2 * c.gid] = res;
  out[2 * c.gid + 1] = ires;
  uint32_t* cyc = reinterpret_cast<uint32_t*>(arg->cyc_addr);
  if (c.lead) { cyc[2 * c.warp] = t0; cyc[2 * c.warp + 1] = t1; }
}

__kernel void alu_ind(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  uint32_t b = c.gid | 1u;
  uint32_t a0 = c.gid, a1 = c.gid + 1, a2 = c.gid + 2, a3 = c.gid + 3, a4 = c.gid + 4, a5 = c.gid + 5, a6 = c.gid + 6,
           a7 = c.gid + 7, a8 = c.gid + 8, a9 = c.gid + 9, a10 = c.gid + 10, a11 = c.gid + 11, a12 = c.gid + 12,
           a13 = c.gid + 13, a14 = c.gid + 14, a15 = c.gid + 15;
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    ADD(a0); ADD(a1); ADD(a2); ADD(a3); ADD(a4); ADD(a5); ADD(a6); ADD(a7);
    ADD(a8); ADD(a9); ADD(a10); ADD(a11); ADD(a12); ADD(a13); ADD(a14); ADD(a15);
    ADD(a0); ADD(a1); ADD(a2); ADD(a3); ADD(a4); ADD(a5); ADD(a6); ADD(a7);
    ADD(a8); ADD(a9); ADD(a10); ADD(a11); ADD(a12); ADD(a13); ADD(a14); ADD(a15);
  }
  uint32_t res = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)) + (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15)));
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emit(arg, c, t0, t1, res);
}

__kernel void alu_dep(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  uint32_t b = c.gid | 1u;
  uint32_t a0 = c.gid;
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0);
    ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0);
    ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0);
    ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0); ADD(a0);
  }
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emit(arg, c, t0, t1, a0);
}

// FP helpers: chain k starts at a0 + 0.1 k as in ps_memprobe fma, a0 = 1 + (gid & 1023) / 1024 (exact)
#define FINIT                                                                                                  \
  float fb = 0.999f, fc = 1.0e-3f;                                                                             \
  __asm__ volatile("" : "+f"(fb), "+f"(fc));                                                                   \
  float x0 = 1.0f + (float)(c.gid & 1023u) * 0.0009765625f;                                                    \
  [[maybe_unused]] float f0 = x0, f1 = x0 + 0.1f, f2 = x0 + 0.2f, f3 = x0 + 0.3f, f4 = x0 + 0.4f, f5 = x0 + 0.5f, f6 = x0 + 0.6f,  \
        f7 = x0 + 0.7f, f8 = x0 + 0.8f, f9 = x0 + 0.9f, f10 = x0 + 1.0f, f11 = x0 + 1.1f, f12 = x0 + 1.2f,     \
        f13 = x0 + 1.3f, f14 = x0 + 1.4f, f15 = x0 + 1.5f;

__kernel void fma_ind16(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  FINIT
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f4); FMA(f5); FMA(f6); FMA(f7);
    FMA(f8); FMA(f9); FMA(f10); FMA(f11); FMA(f12); FMA(f13); FMA(f14); FMA(f15);
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f4); FMA(f5); FMA(f6); FMA(f7);
    FMA(f8); FMA(f9); FMA(f10); FMA(f11); FMA(f12); FMA(f13); FMA(f14); FMA(f15);
  }
  float r = (((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7))) + (((f8 + f9) + (f10 + f11)) + ((f12 + f13) + (f14 + f15)));
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emitf(arg, c, t0, t1, r, 0);
}

__kernel void fma_ind8(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  FINIT
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f4); FMA(f5); FMA(f6); FMA(f7);
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f4); FMA(f5); FMA(f6); FMA(f7);
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f4); FMA(f5); FMA(f6); FMA(f7);
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f4); FMA(f5); FMA(f6); FMA(f7);
  }
  float r = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7));
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emitf(arg, c, t0, t1, r, 0);
}

__kernel void fma_ind4(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  FINIT
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f0); FMA(f1); FMA(f2); FMA(f3);
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f0); FMA(f1); FMA(f2); FMA(f3);
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f0); FMA(f1); FMA(f2); FMA(f3);
    FMA(f0); FMA(f1); FMA(f2); FMA(f3); FMA(f0); FMA(f1); FMA(f2); FMA(f3);
  }
  float r = (f0 + f1) + (f2 + f3);
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emitf(arg, c, t0, t1, r, 0);
}

__kernel void fma_dep(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  FINIT
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0);
    FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0);
    FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0);
    FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0); FMA(f0);
  }
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emitf(arg, c, t0, t1, f0, 0);
}

__kernel void fadd_ind(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  FINIT
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    FADD(f0); FADD(f1); FADD(f2); FADD(f3); FADD(f4); FADD(f5); FADD(f6); FADD(f7);
    FADD(f0); FADD(f1); FADD(f2); FADD(f3); FADD(f4); FADD(f5); FADD(f6); FADD(f7);
    FADD(f0); FADD(f1); FADD(f2); FADD(f3); FADD(f4); FADD(f5); FADD(f6); FADD(f7);
    FADD(f0); FADD(f1); FADD(f2); FADD(f3); FADD(f4); FADD(f5); FADD(f6); FADD(f7);
  }
  float r = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7));
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emitf(arg, c, t0, t1, r, 0);
}

__kernel void fsgnj_ind(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  FINIT
  fb = -fb;      // sign source negative: every chain ends up with a negative sign, magnitude unchanged
  __asm__ volatile("" : "+f"(fb));
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    FSGN(f0); FSGN(f1); FSGN(f2); FSGN(f3); FSGN(f4); FSGN(f5); FSGN(f6); FSGN(f7);
    FSGN(f0); FSGN(f1); FSGN(f2); FSGN(f3); FSGN(f4); FSGN(f5); FSGN(f6); FSGN(f7);
    FSGN(f0); FSGN(f1); FSGN(f2); FSGN(f3); FSGN(f4); FSGN(f5); FSGN(f6); FSGN(f7);
    FSGN(f0); FSGN(f1); FSGN(f2); FSGN(f3); FSGN(f4); FSGN(f5); FSGN(f6); FSGN(f7);
  }
  float r = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7));
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emitf(arg, c, t0, t1, r, 0);
}

__kernel void fcvt_ind(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  uint32_t b = c.gid | 1u;
  float f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0, f6 = 0, f7 = 0;
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    FCVT(f0); FCVT(f1); FCVT(f2); FCVT(f3); FCVT(f4); FCVT(f5); FCVT(f6); FCVT(f7);
    FCVT(f0); FCVT(f1); FCVT(f2); FCVT(f3); FCVT(f4); FCVT(f5); FCVT(f6); FCVT(f7);
    FCVT(f0); FCVT(f1); FCVT(f2); FCVT(f3); FCVT(f4); FCVT(f5); FCVT(f6); FCVT(f7);
    FCVT(f0); FCVT(f1); FCVT(f2); FCVT(f3); FCVT(f4); FCVT(f5); FCVT(f6); FCVT(f7);
  }
  float r = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7));
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emitf(arg, c, t0, t1, r, 0);
}

__kernel void mix_fa(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  FINIT
  uint32_t b = c.gid | 1u;
  uint32_t a0 = c.gid, a1 = c.gid + 1, a2 = c.gid + 2, a3 = c.gid + 3, a4 = c.gid + 4, a5 = c.gid + 5, a6 = c.gid + 6,
           a7 = c.gid + 7;
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    FMA(f0); ADD(a0); FMA(f1); ADD(a1); FMA(f2); ADD(a2); FMA(f3); ADD(a3);
    FMA(f4); ADD(a4); FMA(f5); ADD(a5); FMA(f6); ADD(a6); FMA(f7); ADD(a7);
    FMA(f0); ADD(a0); FMA(f1); ADD(a1); FMA(f2); ADD(a2); FMA(f3); ADD(a3);
    FMA(f4); ADD(a4); FMA(f5); ADD(a5); FMA(f6); ADD(a6); FMA(f7); ADD(a7);
  }
  float r = ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7));
  uint32_t ires = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emitf(arg, c, t0, t1, r, ires);
}

__kernel void ld_ind(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  const char* p = reinterpret_cast<const char*>(arg->buf_addr) + 4u * c.gid;   // coalesced: lane = consecutive word
  uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = r10 = r11 = r12 = r13 = r14 = r15 = 0;
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    LD(r0, 0); LD(r1, 1); LD(r2, 2); LD(r3, 3); LD(r4, 4); LD(r5, 5); LD(r6, 6); LD(r7, 7);
    LD(r8, 0); LD(r9, 1); LD(r10, 2); LD(r11, 3); LD(r12, 4); LD(r13, 5); LD(r14, 6); LD(r15, 7);
    LD(r0, 0); LD(r1, 1); LD(r2, 2); LD(r3, 3); LD(r4, 4); LD(r5, 5); LD(r6, 6); LD(r7, 7);
    LD(r8, 0); LD(r9, 1); LD(r10, 2); LD(r11, 3); LD(r12, 4); LD(r13, 5); LD(r14, 6); LD(r15, 7);
  }
  uint32_t res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)) + (((r8 + r9) + (r10 + r11)) + ((r12 + r13) + (r14 + r15)));
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emit(arg, c, t0, t1, res);
}

__kernel void ld_dep(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  uint32_t pv = arg->buf_addr + 4u * c.gid;     // cell holds its own address: every load is an L1 hit after the first
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
#define DL() __asm__ volatile("lw %0, 0(%0)" : "+r"(pv))
    DL(); DL(); DL(); DL(); DL(); DL(); DL(); DL();
    DL(); DL(); DL(); DL(); DL(); DL(); DL(); DL();
    DL(); DL(); DL(); DL(); DL(); DL(); DL(); DL();
    DL(); DL(); DL(); DL(); DL(); DL(); DL(); DL();
  }
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emit(arg, c, t0, t1, pv);
}

__kernel void br8(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  uint32_t b = c.gid | 1u;
  uint32_t a0 = c.gid, a1 = c.gid + 1, a2 = c.gid + 2, a3 = c.gid + 3, a4 = c.gid + 4, a5 = c.gid + 5, a6 = c.gid + 6;
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    ADD(a0); ADD(a1); ADD(a2); ADD(a3); ADD(a4); ADD(a5); ADD(a6); JMP();
    ADD(a0); ADD(a1); ADD(a2); ADD(a3); ADD(a4); ADD(a5); ADD(a6); JMP();
    ADD(a0); ADD(a1); ADD(a2); ADD(a3); ADD(a4); ADD(a5); ADD(a6); JMP();
    ADD(a0); ADD(a1); ADD(a2); ADD(a3); ADD(a4); ADD(a5); ADD(a6); JMP();
  }
  uint32_t res = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + a6);
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emit(arg, c, t0, t1, res);
}

__kernel void br4(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  uint32_t b = c.gid | 1u;
  uint32_t a0 = c.gid, a1 = c.gid + 1, a2 = c.gid + 2;
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < c.n; ++i) {
    ADD(a0); ADD(a1); ADD(a2); JMP(); ADD(a0); ADD(a1); ADD(a2); JMP();
    ADD(a0); ADD(a1); ADD(a2); JMP(); ADD(a0); ADD(a1); ADD(a2); JMP();
    ADD(a0); ADD(a1); ADD(a2); JMP(); ADD(a0); ADD(a1); ADD(a2); JMP();
    ADD(a0); ADD(a1); ADD(a2); JMP(); ADD(a0); ADD(a1); ADD(a2); JMP();
  }
  uint32_t res = (a0 + a1) + a2;
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  emit(arg, c, t0, t1, res);
}

// ffl_chk: functional check of the FFLAGS patch. csrw fflags,0; fdiv 1/0 (DZ = 0x08); fmul 3e38*3e38 (OF|NX = 0x05); fdiv 1/3 (NX = 0x01);
// then csrr fflags IMMEDIATELY. Correct (drained) result = 0x0D in every thread. With the stock scoreboard the csrr waits via the FFLAGS WAW bit;
// with the patch it must wait for fp_inflight_ == 0. A missing drain would read 0 (the flags of in-flight FP ops not yet accumulated).
__kernel void ffl_chk(ps_issueprobe_arg_t* __UNIFORM__ arg) {
  Ctx c = ctx(arg);
  float one = 1.0f, zero = 0.0f, three = 3.0f, big = 3.0e38f, r0, r1, r2;
  __asm__ volatile("" : "+f"(one), "+f"(zero), "+f"(three), "+f"(big));
  uint32_t fl;
  __asm__ volatile("csrw fflags, x0");
  __asm__ volatile("fdiv.s %0, %1, %2" : "=f"(r0) : "f"(one), "f"(zero));
  __asm__ volatile("fmul.s %0, %1, %2" : "=f"(r1) : "f"(big), "f"(big));
  __asm__ volatile("fdiv.s %0, %1, %2" : "=f"(r2) : "f"(one), "f"(three));
  __asm__ volatile("csrr %0, fflags" : "=r"(fl));
  emit(arg, c, 0, 0, fl);
  __asm__ volatile("" : : "f"(r0), "f"(r1), "f"(r2));
}
