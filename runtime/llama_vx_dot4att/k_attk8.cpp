// k_attk8.cpp -- ps_llamadot4att: attn_k8_dot / attn_k8_sw of kernels/ps_attn_k8 (decode attention with an int8 K cache, QK^T by DOT4ACC or scalar int8,
// q and the newest K row quantized in the kernel prologue; V, softmax fp32), tree psdot/psall/psall2 only for attn_k8_dot. Own translation unit, linked last, so
// that every other launch of the app is unchanged (see kernel.cpp for why each reused kernel is its own translation unit).
#include "reused/ps_attn_k8/kernel_k8.cpp"
