// common.h -- ps_scan_micro kernel arguments (host main.cpp and device kernel.cpp). (copy of
// ps_wkv_micro with warm-up passes; ps_wkv_micro itself is unchanged so the earlier evidence keeps its binary).
#ifndef PS_SCAN_MICRO_COMMON_H
#define PS_SCAN_MICRO_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t w_addr;     // weight block: w_vec uint4 vectors (16 B each), re-read every round
  uint64_t kv_addr;    // KV stream: rounds * kv_vec uint4 vectors, round r reads [r * kv_vec, (r + 1) * kv_vec)
  uint64_t out_addr;   // one uint32 per thread: wrap-around sum of every 32-bit word the thread read
  uint32_t w_vec;
  uint32_t kv_vec;
  uint32_t rounds;
  uint32_t warm;       // extra weight-block passes before round 0 (the re-used set's first re-reference)
} ps_scan_arg_t;

#endif
