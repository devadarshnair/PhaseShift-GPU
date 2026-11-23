// k_dot4w.cpp -- ps_llamadot3: the warp-cooperative DOT4ACC Q8_0 GEMV of kernels/ps_gemv_q80dot4w (entries gemv_q80_w<S>u<U>, tree psdot/psall/psall2 only).
// Weights must be in the host re-layout of kernels/ps_gemv_q80dot4w/common.h; ps_llamadot3 does that once at load time for every matrix selected by -S.
#include "reused/gemv_q80dot4w/kernel.cpp"
