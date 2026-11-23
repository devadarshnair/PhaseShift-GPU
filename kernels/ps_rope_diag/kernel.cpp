// kernel.cpp -- ps_rope_diag device code: the UNCHANGED ps_rope device kernel, compiled from its own source.
// diagnoses the ps_rope seed-1 failure; this app reuses rope_kv verbatim so its machine code equals ps_rope's
// (checked by llvm-objdump diff in results/<run>/), while the diagnostic host options live in this app's main.cpp and
// ps_rope's own sources (and so vxrun's app-source hash and every stored ps_rope run key) stay untouched.
#include "rope_kernel.cpp"  // symlink to ../rope/kernel.cpp (hashed by vxrun; DEPS in app.conf rebuilds on change)
