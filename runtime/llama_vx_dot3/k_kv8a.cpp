// k_kv8a.cpp -- ps_llama device module, translation unit 12 of 12: attn_gqa_kv8 / attn_dxa_kv8 (and attn_*_kv16 when the build has Zfhmin) of
// kernels/decode_attn_gqa/kernel_kv.cpp, the decode attention over the int8 KV cache; ps_llama -Q kv8 uses attn_gqa_kv8. Linked last (see k_kv8r.cpp).
#include "reused/decode_attn_gqa/kernel_kv.cpp"
