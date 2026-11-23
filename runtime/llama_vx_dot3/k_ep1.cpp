// k_ep1.cpp -- ps_llama device module, translation unit 10 of 10: the element-wise repair entries rope_table, rope_kv_tab
// (kernels/rope/kernel_tab.cpp) and silu_gate_expf_poly, silu_gate_expf_mufu (kernels/silu_gate/kernel_exp.cpp), used by
// `ps_llama -R table` and `-E poly|mufu`. They sit in a translation unit of their own, linked last, so that the code of the nine
// earlier units (and with it every default-mode launch) is unchanged by this step (see kernel.cpp for why each reused kernel is
// its own translation unit; the two files are included in ONE unit because they share no file-local names).
#include "reused/rope/kernel_tab.cpp"
#include "reused/silu_gate/kernel_exp.cpp"
