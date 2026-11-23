// k_linear.cpp -- ps_llama device module, translation unit 7 of 9: the small-M fp32 GEMM (entry gemm_small; the TCU entry is compiled only with -DVX_CFG_TCU_WGMMA_ENABLE), used by the batched-prefill mode, unchanged
// (see kernel.cpp for why each reused kernel is its own translation unit).
#include "reused/linear/kernel.cpp"
