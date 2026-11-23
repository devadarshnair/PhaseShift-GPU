// kernel.cpp -- ps_wkv_micro device kernel (weight block ~1/2 L2 re-read inside one launch + streaming
// KV). Each of `rounds` rounds reads the whole weight block once and then a fresh, never-reused KV chunk, both with a
// grid-stride loop over 16-byte vectors (consecutive lanes read consecutive vectors, so a warp load covers
// NT x 16 B = 8 lines at NT = 32). The weight block is therefore re-referenced once per round with a reuse distance
// of (weight + KV chunk) bytes: with KV chunk + weight > L2 capacity, MRU insertion of the KV fills evicts the block
// before its next round (the thrashing case TEST), while inserting the KV fills at LRU (or not
// allocating them) leaves the block resident. Integer wrap-around sums make the host check exact (no tolerance).
#include <vx_spawn2.h>
#include "common.h"

struct u4 { uint32_t x, y, z, w; };

__kernel void wkv(ps_wkv_arg_t* __UNIFORM__ arg) {
  const u4* W = reinterpret_cast<const u4*>(arg->w_addr);
  const u4* KV = reinterpret_cast<const u4*>(arg->kv_addr);
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out_addr);
  const uint32_t nthr = gridDim.x * blockDim.x;
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t wv = arg->w_vec, kvv = arg->kv_vec, rounds = arg->rounds;
  uint32_t acc = 0;
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
