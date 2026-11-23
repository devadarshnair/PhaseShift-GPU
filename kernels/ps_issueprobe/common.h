// common.h -- ps_issueprobe kernel arguments (per-warp issue-cadence microbenchmarks).
#ifndef PS_ISSUEPROBE_COMMON_H
#define PS_ISSUEPROBE_COMMON_H

#include <stdint.h>

typedef struct {
  uint64_t out_addr;   // per thread: {result word, unused}  (2 words per global thread id)
  uint64_t cyc_addr;   // per warp:   {mcycle before the loop, mcycle after the loop}  (2 words per warp)
  uint64_t buf_addr;   // load variants: source buffer (8 KiB)
  uint32_t iters;      // loop iterations n
  uint32_t pad;
} ps_issueprobe_arg_t;

#endif
