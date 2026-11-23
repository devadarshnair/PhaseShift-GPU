// kernel.cpp -- ps_memprobe device kernels: measured machine parameters of a PS-BASE configuration.
// New microbenchmark code (not one of the validated P2 kernels).
//   chase  : ONE thread follows a pointer chain (node = 64 B line, first word = byte offset of the next node). Every
//            load's address depends on the previous load, so the timed loop measures the load-to-use latency of the
//            level the footprint lives in, plus one integer add per hop (the address add). The cycle counter
//            (mcycle CSR) is read before and after the timed hops; the final offset is stored BEFORE the second read,
//            so the store waits for the last load and no hop is outstanding when the counter is read.
//   stream : all threads read a word array with a grid stride (consecutive lanes -> consecutive words, i.e. a warp
//            load covers NT*4 contiguous bytes) and sum it as uint32 (exact, host-checked). 4 independent loads per
//            iteration so each warp keeps up to 4 loads in flight (memory-level parallelism); DRAM bytes/cycle =
//            bytes read / kernel cycles.
//   fma    : every thread runs 8 independent fused multiply-add chains (a = a*b + c) for `steps` iterations; the
//            8 chains hide the FPU latency within a warp, so FLOP/cycle approaches the FPU issue limit.
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

__kernel void chase(ps_memprobe_arg_t* __UNIFORM__ arg) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const char* base = reinterpret_cast<const char*>(arg->buf_addr);
  volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(arg->out_addr);
  uint32_t p = arg->start;
  uint32_t t0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < arg->warm; ++i) p = *reinterpret_cast<const uint32_t*>(base + p);
  out[3] = p;  // waits for the last warm-up load
  uint32_t t1 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < arg->steps; ++i) p = *reinterpret_cast<const uint32_t*>(base + p);
  out[0] = p;  // waits for the last timed load
  uint32_t t2 = csr_read(VX_CSR_MCYCLE);
  out[1] = t2 - t1;
  out[2] = t1 - t0;
}

__kernel void stream(ps_memprobe_arg_t* __UNIFORM__ arg) {
  const uint32_t* src = reinterpret_cast<const uint32_t*>(arg->buf_addr);
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out_addr);
  const uint32_t n = arg->steps;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t j = gid;
  for (; j + 3 * stride < n; j += 4 * stride) {
    uint32_t v0 = src[j], v1 = src[j + stride], v2 = src[j + 2 * stride], v3 = src[j + 3 * stride];
    a0 += v0; a1 += v1; a2 += v2; a3 += v3;
  }
  for (; j < n; j += stride) a0 += src[j];
  out[gid] = a0 + a1 + a2 + a3;
}

__kernel void fma(ps_memprobe_arg_t* __UNIFORM__ arg) {
  float* out = reinterpret_cast<float*>(arg->out_addr);
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  float b = 0.999f, c = 1.0e-3f;
  const uint32_t n = arg->steps;
  // keep b and c in registers: without this the compiler re-materialises them with two flw per iteration and the
  // fmas then wait on a load every iteration (measured in development: ~14 FLOP/cycle/4 cores lower)
  __asm__ volatile("" : "+f"(b), "+f"(c));
  float a0 = 1.0f + (float)(gid & 1023u) * 0.0009765625f,  // exact (no rounding, so FMA contraction is harmless)
        a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
  float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
  for (uint32_t i = 0; i < n; ++i) {
    a0 = __builtin_fmaf(a0, b, c); a1 = __builtin_fmaf(a1, b, c); a2 = __builtin_fmaf(a2, b, c);
    a3 = __builtin_fmaf(a3, b, c); a4 = __builtin_fmaf(a4, b, c); a5 = __builtin_fmaf(a5, b, c);
    a6 = __builtin_fmaf(a6, b, c); a7 = __builtin_fmaf(a7, b, c);
  }
  out[gid] = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}
