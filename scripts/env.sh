# PhaseShift shell environment. Usage: `source phaseshift/scripts/env.sh` (bash)
# Every shell step starts with this. It only sets variables; it never installs, builds or starts anything.

export PS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TOOLDIR="$HOME/tools"

# Build directories: base = infra patch only, ps = infra + mechanism patches.
# build/vortex-upstream is deprecated and must never be built in (its VORTEX_HOME is the pristine clone).
export VX_BASE="$PS/build/vortex-base"
export VX_PS="$PS/build/vortex-ps"

# Prebuilt EDA tools from the Vortex toolchain install (~/tools). Prefix only; avoid duplicating on re-source.
for _ps_d in sv2v sta yosys verilator; do
    case ":$PATH:" in
        *":$TOOLDIR/$_ps_d/bin:"*) ;;
        *) PATH="$TOOLDIR/$_ps_d/bin:$PATH" ;;
    esac
done
unset _ps_d
export PATH

# Low-load machine: single-threaded builds and simulators.
#   MAKEFLAGS=-j1  -> make never runs parallel recipes (third_party/Makefile still hard-codes -j4; see errata 22).
#   THREADS=1      -> rtlsim's Makefile (sim/rtlsim/Makefile:165-166) otherwise defaults to all cores for Verilator.
#   *_NUM_THREADS  -> numpy/torch BLAS pools single-threaded.
export MAKEFLAGS=-j1
export THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# ccache shared by both trees: CCACHE_BASEDIR rewrites absolute paths under $PS to relative ones, so identical
# sources in vortex-trees/base and vortex-trees/ps hit the same cache entries.
export CCACHE_DIR="$PS/build/ccache"
export CCACHE_MAXSIZE=5G
export CCACHE_BASEDIR="$PS"

# cmake 4.x rejects cmake_minimum_required < 3.5 in Ramulator's FetchContent deps. cmake honours
# this as an environment variable (errata 19); the infra patch also passes it as -D.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# /tmp is a 3.8 GB RAM-backed tmpfs (errata 21): compiler temporaries and scratch go to disk.
export TMPDIR="$PS/build/tmp"
mkdir -p "$TMPDIR"

# Heavy-job wrapper (see scripts/lowload.sh header for the contract).
export LL="$PS/scripts/lowload.sh"

# Project Python: the phaseshift conda env's interpreter, for scripts that must not depend on `conda run`.
export PY="$HOME/miniconda3/envs/phaseshift/bin/python"
