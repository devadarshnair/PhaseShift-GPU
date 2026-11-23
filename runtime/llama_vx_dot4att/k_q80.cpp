// k_q80.cpp -- ps_llama device module, translation unit 9 of 9: the Q8_0 kernels of kernels/gemv_q80
// (gemv_q80_rows, gemv_q80_lanes, quantize_q80), unchanged, used by `ps_llama -q q80` (see kernel.cpp for why each reused
// kernel is its own translation unit; the other eight units are untouched).
#include "reused/gemv_q80/kernel.cpp"
