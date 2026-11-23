// main.cpp -- ps_gemv_f32 host app: y = W @ x on Vortex, checked against a ps-fixture-v1 fixture
// or a synthetic case. All logic lives in gemv_host.h (shared with ps_twolaunch); default: 1 launch.
#include "gemv_host.h"

int main(int argc, char** argv) { return ps_gemv::run(argc, argv, 1); }
