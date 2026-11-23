// common.h -- ps_phase_rw kernel arguments, shared by host main.cpp and device kernel.cpp.
#ifndef PS_PHASE_RW_COMMON_H
#define PS_PHASE_RW_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t x_addr;  // n uint32
  uint64_t y_addr;  // n uint32
  uint32_t n;
  uint32_t a;       // y[i] = x[i] * a + b (mod 2^32), different per launch so every launch's output is distinct
  uint32_t b;
} ps_phase_rw_arg_t;

#endif
