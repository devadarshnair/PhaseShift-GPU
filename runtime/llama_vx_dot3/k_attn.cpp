// k_attn.cpp -- ps_llama device module, translation unit 6 of 9: the fused GQA decode attention (entries attn_gqa, attn_naive, attn_dxa; ps_llama uses attn_gqa), unchanged
// (see kernel.cpp for why each reused kernel is its own translation unit).
#include "reused/decode_attn_gqa/kernel.cpp"
