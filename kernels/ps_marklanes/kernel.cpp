// kernel.cpp -- ps_marklanes device kernel (carry-over 3). A TEST app, not an validated
// kernel: it writes VX_CSR_PS_MARK from MANY lanes on purpose (PS_MARK without the `if (tid == 0)` guard the validated
// kernels use), to check that SimX counts exactly one work unit per warp CSR-write instruction when NUM_SFU_LANES <
// NUM_THREADS (the SFU dispatcher then splits the instruction into one trace per SFU packet with an active lane;
// patch ps-counters counts only the first trace, trace->sop; errata 33b). Build with -DPS_HINTS=1 (otherwise
// PS_MARK compiles to nothing and the count is 0).
//   mode 0: every lane marks             -> the instruction is split into NT / SFU_LANES packet traces
//   mode 1: only lane NT-1 marks         -> one trace, in the LAST packet (sop is set on it)
//   mode 2: lanes NT/2 .. NT-1 mark      -> traces of the upper packets only
// In every mode each warp executes `reps` mark instructions, so the expected count is (#warps) * reps.
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"
#include "ps_hint.h"

__kernel void marklanes(ps_marklanes_arg_t* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out_addr);
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = (uint32_t)vx_thread_id();
  const uint32_t nt = (uint32_t)vx_num_threads();
  for (uint32_t r = 0; r < arg->reps; ++r) {
    if (arg->mode == 0) {
      PS_MARK(PS_MARK_OTHER);
    } else if (arg->mode == 1) {
      if (lane == nt - 1) PS_MARK(PS_MARK_PREFILL_TILE);
    } else {
      if (lane >= nt / 2) PS_MARK(PS_MARK_DECODE_TILE);
    }
  }
  out[gid] = gid + 1;
}
