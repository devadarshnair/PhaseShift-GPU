// kernel.cpp -- ps_attn_k8 device module, translation unit 1 of 2: the stock fp32 fused GQA decode attention (attn_gqa) of kernels/decode_attn_gqa, UNCHANGED,
// linked in the same module as kernel_k8.cpp so that the fp32 reference arm and the int8 arms of the microbench run the same harness.
#include "reused/decode_attn_gqa/kernel.cpp"
