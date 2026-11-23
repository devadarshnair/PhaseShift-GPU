// common.h -- ps_memprobe kernel arguments (machine-parameter microbenchmarks).
#ifndef PS_MEMPROBE_COMMON_H
#define PS_MEMPROBE_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t buf_addr;   // chase: node array (each node's first word = byte offset of the next node); stream: source
  uint64_t out_addr;   // chase: {final offset, cycles of the timed hops, cycles of the warm-up hops}; stream/fma: per thread
  uint32_t start;      // chase: byte offset of the first node
  uint32_t warm;       // chase: untimed warm-up hops (fill L1/L2 with the footprint); fma: unused
  uint32_t steps;      // chase: timed hops; stream: words in the source; fma: loop iterations per thread
  uint32_t pad;
} ps_memprobe_arg_t;

#endif
