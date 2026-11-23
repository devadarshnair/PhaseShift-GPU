// common.h -- ps_wkv_micro kernel arguments (host main.cpp and device kernel.cpp).
#ifndef PS_WKV_MICRO_COMMON_H
#define PS_WKV_MICRO_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t w_addr;     // weight block: w_vec uint4 vectors (16 B each), re-read every round
  uint64_t kv_addr;    // KV stream: rounds * kv_vec uint4 vectors, round r reads [r * kv_vec, (r + 1) * kv_vec)
  uint64_t out_addr;   // one uint32 per thread: wrap-around sum of every 32-bit word the thread read
  uint32_t w_vec;
  uint32_t kv_vec;
  uint32_t rounds;
  uint32_t pad;
} ps_wkv_arg_t;

#endif
