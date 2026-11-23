// common.h -- ps_mix_micro kernel arguments (host main.cpp and device kernel.cpp).
#ifndef PS_MIX_MICRO_COMMON_H
#define PS_MIX_MICRO_COMMON_H

#include <stdint.h>

#define PS_MIX_DXA_DESC 0     // DXA descriptor slot of the bulk buffer (1D, 4-byte elements)
#define PS_MIX_MAX_DEPTH 4    // DXA tiles in flight per CTA (barrier ids 1..depth; id 0 is __syncthreads' slot)

typedef struct {
  uint64_t a_addr;      // GEMV matrix, rows x cols uint32 (row-major), read once (decode weight stream)
  uint64_t x_addr;      // GEMV vector, cols uint32
  uint64_t part_addr;   // rows x NT uint32: lane partial dot products (wrap-around), exact host check
  uint64_t bulk_addr;   // bulk buffer, bulk_tiles x tile_words uint32, copied once to LMEM by DXA
  uint64_t bsum_addr;   // grid x NT uint32: per-lane wrap-around sum of the words the bulk warp found in LMEM
  uint64_t cyc_addr;    // grid x NW x 4 uint32: per warp {start lo, start hi, end lo, end hi} (mcycle)
  uint32_t rows;
  uint32_t cols;        // multiple of 4 (uint4 loads)
  uint32_t bulk_tiles;  // tiles in the bulk buffer (0 = no bulk traffic)
  uint32_t tile_words;  // words per DXA tile (multiple of NT)
  uint32_t depth;       // DXA tiles in flight per CTA, 1..PS_MIX_MAX_DEPTH
  uint32_t gemv_on;     // 0 = the GEMV warps do nothing (bulk-only reference)
} ps_mix_arg_t;

#endif
