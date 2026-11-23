// k_kv8r.cpp -- ps_llama device module, translation unit 11 of 12: rope_kv_tab_q8 (kernels/rope/kernel_kv8.cpp), RoPE from the angle table with an
// int8 KV-cache write, used by `ps_llama -q q80 -Q kv8`. Its own translation unit, linked after the ten earlier ones, so that their code (and with it every
// default-mode launch) is unchanged; it is not merged with k_kv8a.cpp because kernel_kv8.cpp compiles with fp contract off (the NumPy port's separate roundings)
// and that pragma must not reach the attention code.
#include "reused/rope/kernel_kv8.cpp"
