// common.h -- ps_marklanes kernel arguments (rev 2.8 carry-over 3), shared by main.cpp and kernel.cpp.
#ifndef PS_MARKLANES_COMMON_H
#define PS_MARKLANES_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t out_addr;  // grid * block uint32: out[i] = i + 1 (a checkable result)
  uint32_t reps;      // marks per thread (loop)
  uint32_t mode;      // 0 = every lane marks (unguarded), 1 = only the warp's last lane, 2 = the upper half of the lanes
} ps_marklanes_arg_t;

#endif
