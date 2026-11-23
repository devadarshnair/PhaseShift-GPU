// main.cpp -- ps_twolaunch host app (cold-cache check). Reuses the ps_gemv_f32 kernel and host driver
// (kernel.cpp, common.h and gemv_host.h are symlinks to ../gemv_f32/, so vxrun's app-source hash of this directory
// covers their contents). Default: 2 back-to-back launches of the same GEMV on the same device buffers in one app
// run; `-L 1` gives the 1-launch reference. The MPM counters accumulate from device reset, not per launch
// (sim/simx/mem/cache.cpp on_reset(): "perf_stats_ survives"), so the L2 read misses printed at the end of a
// 2-launch run are the sum over both launches. If SimX kept cache contents across launches, the second launch
// would hit in L2 (the stories15M wq matrix, 288 x 288 x 4 B = 324 KiB, fits the 1 MiB default L2) and
// misses(2 launches) would be well below 2 x misses(1 launch).
#include "gemv_host.h"

int main(int argc, char** argv) { return ps_gemv::run(argc, argv, 2); }
