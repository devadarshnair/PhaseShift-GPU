// kernel.cpp -- ps_llama device module, translation unit 1 of 9: the GEMV kernels gemv_rows /
// gemv_lanes, unchanged.
//
// ps_llama loads ONE multi-entry .vxbin (the tests/regression/multikernel pattern: every `__kernel` of every linked
// object is a named entry, resolved by vx_module_get_kernel; a module image is linked at a fixed address, so one module
// per session, cf. tests/regression/module_reload). The entries come from the validated standalone kernels, each
// compiled in its own translation unit so that their file-local helpers (bits_from_f32, warp_ops.h, ...) cannot
// collide: this file and k_*.cpp each #include one kernel's kernel.cpp through per-file symlinks
// (reused/<k>/<file> -> ../../../../kernels/<k>/<file>, for kernel.cpp, common.h and warp_ops.h -- exactly the files the
// device code and the host's argument structs use; the quoted includes inside kernel.cpp resolve in reused/<k>/), so the
// device code is byte-for-byte the validated source and vxrun's app-source hash (rglob over kernels/llama ->
// runtime/llama_vx, file symlinks followed) covers precisely these files. app.conf lists the translation units
// (VX_SRCS) and the included sources as make prerequisites (DEPS).
#include "reused/gemv_f32/kernel.cpp"
