// k_pattn.cpp -- ps_llama device module, translation unit 8 of 9: the tiled causal prefill attention (entries attn_prefill_lmem, attn_prefill_dxa; ps_llama uses attn_prefill_lmem), used by the batched-prefill mode, unchanged
// (see kernel.cpp for why each reused kernel is its own translation unit).
#include "reused/prefill_attn/kernel.cpp"
