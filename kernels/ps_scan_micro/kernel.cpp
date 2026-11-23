// kernel.cpp -- ps_scan_micro device kernel (scan-resistance microbenchmark). = ps_wkv_micro's kernel
// plus `warm` extra passes over the weight block before round 0. Why the warm-up: a re-reference-interval
// policy (SRRIP [C34]) protects a block only after the block has been re-referenced once while resident (a hit sets
// its RRPV to 0). In ps_wkv_micro every weight re-reference comes after a KV chunk of >= L2 - weight bytes, so under
// SRRIP no weight line is ever hit, every fill is inserted at the same RRPV and the victim order degenerates to the
// stock one (measured: SRRIP cycles identical to stock on -w 512 -c 1024 -r 16, results/<run>). One warm-up pass
// (weight block read twice back to back; it fits the L2) gives the block its first hit; then the rounds interleave
// the re-used block with the one-pass scan (rounds x KV chunk, each byte read once), as the TEST specifies.
// Under LRU the warm-up hit does not help: the next KV chunk still evicts the block before its next round.
#include <vx_spawn2.h>
#include "common.h"

struct u4 { uint32_t x, y, z, w; };

__kernel void scan(ps_scan_arg_t* __UNIFORM__ arg) {
  const u4* W = reinterpret_cast<const u4*>(arg->w_addr);
  const u4* KV = reinterpret_cast<const u4*>(arg->kv_addr);
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out_addr);
  const uint32_t nthr = gridDim.x * blockDim.x;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t wv = arg->w_vec, kvv = arg->kv_vec, rounds = arg->rounds;
  uint32_t acc = 0;
  for (uint32_t p = 0; p < arg->warm; ++p) {
    for (uint32_t i = gtid; i < wv; i += nthr) {
      u4 a = W[i];
      acc += a.x + a.y + a.z + a.w;
    }
  }
  for (uint32_t r = 0; r < rounds; ++r) {
    for (uint32_t i = gtid; i < wv; i += nthr) {
      u4 a = W[i];
      acc += a.x + a.y + a.z + a.w;
    }
    const u4* K = KV + (size_t)r * kvv;
    for (uint32_t i = gtid; i < kvv; i += nthr) {
      u4 a = K[i];
      acc += a.x + a.y + a.z + a.w;
    }
  }
  out[gtid] = acc;
}
