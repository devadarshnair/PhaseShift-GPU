// main.cpp -- ps_llama: llama2.c full-model driver on Vortex SimX.
//
// Runs run.c's forward pass (llama2.c, run.c `forward`, cited as [cand:D15]) with every op on the device, using only the
// validated standalone kernels (device code unchanged; one multi-entry module, see kernel.cpp):
//   per token and layer l:  rmsnorm -> q, k, v GEMV -> rope_kv (RoPE + KV-cache write) -> decode attention -> o GEMV
//                           -> residual -> rmsnorm -> w1, w3 GEMV -> silu_gate -> w2 GEMV -> residual
//   then final rmsnorm -> classifier GEMV (logits) -> read back -> argmax on the host (run.c `sample` at temperature 0:
//   the first index of the maximum).
// Weights and the KV cache live in device memory for the whole run (uploaded once, before the first launch; host DMA
// writes are outside every kernel's cycle count). The token embedding row is gathered by a device-to-device copy
// (vx_enqueue_copy: the CP's DMA engine, no kernel launch, outside the cycle count) -- run.c's memcpy of
// token_embedding_table[token].
//
// Usage: ps_llama -w <model.bin> -T <golden_tokens.txt> [-m tf|free|prefill] [-n positions] [-v auto|rows|lanes]
//                 [-c b_cap] [-r ref_logits.f32] [-o logits.f32] [-k kernel.vxbin] [-q f32|q80]
//                 [-R perlaunch|table] [-E musl|poly|mufu] [-Q fp32|kv8]
//   -w  llama2.c v0 checkpoint (7 x int32 header, negative vocab_size = unshared classifier, as run.c); read-only.
//       With -q q80: a llama2.c version-2 (Q8_0) export (runq.c `read_checkpoint`: 256-byte header, fp32 norms, then
//       int8 values + fp32 group scales per tensor), e.g. build/models/stories15M_q80.bin
//   -q  f32 (default: every behaviour of this app below is unchanged) | q80 (runq.c's forward on the device):
//       before every matmul the activation vector is quantized by the device kernel quantize_q80 (runq.c `quantize`), then
//       the Q8_0 GEMV gemv_q80_{rows,lanes} multiplies it with the int8 weights (group sums in int32, scales in fp32,
//       runq.c `matmul`). One quantization is shared by the matmuls of one x as in runq.c: q, k, v share xq; w1, w3 share
//       xq; wo, w2 and the classifier quantize their own input (w2's in the `hidden_dim`-long hq). The token embedding
//       is the DEQUANTIZED table row (runq.c:203-205), written by the host like the fp32 copy. Launches per decode token:
//       18 L + 3 (L = 6: 111). tf and free modes only; the logits reference -r is model/ref_q80.py's.
//   -R  (methodology §18.7) RoPE angles: perlaunch (default: rope_kv evaluates powf/cosf/sinf per token, everything of this
//       app unchanged) | table: ONE setup launch `rope_table` before position 0 computes cos/sin for every (pos < seq_len, pair) with
//       the same musl expressions on the device (PSL line pos=-1 op=rope_table: a model-load cost, not part of any PSTOK
//       cycle count), the table is read back and compared entry by entry with a correctly-rounded host recomputation (PSCHECK
//       rope_table_vs_host, PSTABLE; the run FAILS if one entry is off), and every rope launch is `rope_kv_tab` (loads the angles,
//       same rotation and KV-write source lines). The logits are bit-identical to the default mode (tests/test_ep_elementwise.py);
//       works in tf, free and prefill mode (the prefill pass passes B = P rows with positions 0..P-1 to the same kernel).
//   -Q  (methodology §18.9) KV-cache precision: fp32 (default: everything of this app unchanged) | kv8 (track P8K8; needs -q q80, tf / free
//       mode; implies -R table): K and V are stored as INT8 with one fp32 scale per (position, KV head) by the device kernel `rope_kv_tab_q8` (RoPE from the
//       table, K rotated, then runq.c's quantize with group = head_size: scale = max|x| / 127.0f, q = round(x / scale); the PSL op stays `rope_kv`), and
//       the decode attention is `attn_gqa_kv8` (dequantizes in the core right after the load, (float) q * scale; everything else is attn_gqa's). Caches:
//       K8, V8 [layer][b_cap][seq_len][kv_dim] int8 (poisoned with 0x55) and scales [layer][b_cap][H_kv][seq_len] float32 (poisoned with NaN, so a read of an
//       unwritten slot gives NaN logits). Logits reference -r and golden -T: model/ref_q80.py --kv8 (NumPy port of exactly this scheme).
//   -E  exp of the SwiGLU gate: musl (default: entry silu_gate, unchanged) | poly (silu_gate_expf_poly, float-only
//       polynomial) | mufu (silu_gate_expf_mufu, MUFU ex2; refused unless the device has the MUFU extension). poly and mufu are
//       NUMERICS-CHANGING repairs (labelled `PSEP1: ... numerics_changing=1`): the logits differ from the default's in the last bits
//       and are judged by the same token rule below (near-tie accepted) and the 1e-3 norm-wise logits bound.
//   -T  golden token file (lines `pos token next forced`; results/S8/golden_tokens_*.txt): the golden
//       sequence seq[0] = token of line 0, seq[i+1] = next of line i; forced[i] = 1 while seq[i+1] is a prompt token
//   -m  tf (default): teacher forcing -- position p is fed seq[p]; at every non-forced p the device argmax is compared
//         with seq[p+1] (rule, below).
//       free: run.c's generate at temperature 0 -- prompt tokens are forced, afterwards the device's own argmax is fed
//         back; the generated tokens must equal the golden ones up to the first disagreement, which must be a near-tie.
//       prefill: the P prompt tokens (P = 1 + number of forced lines) are processed in ONE batched pass (positions
//         0..P-1 as P rows: gemm_small for the linear layers, prefill attention attn_prefill_lmem for attention, every
//         element-wise kernel with B = P rows), producing the final prompt position's logits; positions P..n-1 then
//         continue in teacher-forced decode mode from the KV cache that the batched pass wrote.: "validated by
//         logits tolerance, not token exactness" -> the final prefill position's logits are compared with the NumPy
//         reference (-r) at rel 1e-3 (the decode positions after it use the token rule as tf).
//   -n  positions (default 32; <= seq_len and <= the golden file's length)
//   -v  GEMV variant per matrix: auto (default; rule "GEMV variant" below), or force rows / lanes for every matrix
//   -c  batch capacity b_cap of the KV-cache layout [layer][b][pos][kv_head][head_dim] (default 1 = run.c's layout,
//       methodology §14.0); the live batch is always B = 1 (sequence 0). b_cap > 1 exercises rope_kv's and the attention
//       kernel's capacity stride (the b_cap > B path).
//   -r  reference logits (float32 [n_ref][vocab], NumPy float32 teacher-forced forward of the golden sequence,
//       model/llama_vx_ref.py): every position whose input history equals the golden one is checked
//   -o  write the device logits of every position (float32 [positions][vocab], little endian) to this file
//
// Output (one line each; parsed by runtime/llama_vx/summarize.py):
//   PSGEMV:  the variant chosen for each weight matrix and why
//   PSL:     one per kernel launch: pos, layer, op, kernel, geometry and the counter deltas of that launch (cycles =
//            max over cores of MCYCLE, instrs = sum over cores of MINSTRET; with VORTEX_PROFILING=1 also scheduler
//            idle and scoreboard stall cycles summed over cores; with VORTEX_PROFILING=7 DRAM reads/writes (64 B
//            requests) and the summed read latency). MPM counters run from device reset (errata 24), so a launch's
//            share is the difference of the counters read before and after it (host reads between launches through
//            the CP's DCR path, outside any kernel's cycle count).
//   PSTOK:   one per position: fed token, device argmax, golden next token, top-1/top-2 logits, the golden token's
//            logit, both relative margins, agree / near-tie flags, the position's cycles and launches
//   PSCHECK: logits_vs_numpy:p<pos> (when -r is given; metric below)
//   PSGEN:   the generated token sequence (free mode)
//   PSSUM:   totals: positions, compared, agreements, near-ties, failures, launches, flushes
//   PERF:    vx_device_dump_perf at exit (cumulative over every launch)
//
// Token rule (with the carry-over): at a compared position the device argmax a
// must equal the golden next token g, OR the disagreement must be an fp32 near-tie in BOTH senses
//     |l1 - l2| / |l1| <= 1e-4   (l1, l2 = top-1 / top-2 device logits: the literal rule)  AND
//     |l1 - lg| / |l1| <= 1e-4   (lg = the device logit of g: the golden token itself must be within the tie margin;
//                                 without this a real bug that pushes g far down could pass whenever the top two of
//                                 the wrong ranking happen to be close).
// Logits metric (modeling choice, methodology §14.5): rel = max_i |l_i - r_i| / max_j |r_j| (norm-wise, L-infinity):
// the logits are one vector whose ranking matters; an element-wise relative error is meaningless for the many logits
// near 0, and the largest logit sets the scale at which a ranking change can happen. Tolerance 1e-3 (for
// the batched-prefill logits; applied to every compared position as a sanity bound -- float32 differences between
// device and NumPy (libm, ps_expf vs np.exp, summation orders) are ~1e-6 relative per op).
//
// GEMV variant (errata 29; modeling choice, methodology §14.5): `rows` (thread per output row) is the
// faster kernel unless the NT rows a warp loads at once map onto too few L1 sets: lane j of a warp reads row
// r0 + j at the same column offset, so with row stride s bytes the lines it touches are (a + (r0 + j) s) / LINE; if
// more than WAYS of them share one set they evict each other before their next float4 reads (errata 29: K = 512 at
// NT = 32 maps 32 rows onto 2 sets, 3.67x slower than an odd-line stride). Rule: rows iff, for every warp-aligned
// r0 and every 16-byte offset a mod LINE, at most DCACHE_NUM_WAYS of the NT rows fall into one L1 set; else lanes
// (warp per row: a warp load is NT consecutive float4, no conflict). A stride of an odd number of lines always passes
// (NT <= sets rows get distinct sets); at REG-CFG (NT = 4, 4 ways) every shape passes. The choice is printed per
// matrix (PSGEMV) and can be forced with -v for the comparison runs.
//
// Kernel launches and cache flushes: every vx_enqueue_launch is followed by one CMD_CACHE_FLUSH that the runtime
// appends itself (sw/runtime/common/device.cpp:816, cp_submit_launch), so flushes = launches; copies, reads, writes
// and DCR reads add none. Per decode token: 14 launches per layer (rmsnorm, q, k, v, rope_kv, attn, o, residual, rmsnorm, w1, w3,
// silu_gate, w2, residual) + final rmsnorm + classifier = 14 L + 2 (72 for stories260K, 86 for stories15M).
#include <unistd.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <vortex.h>   // vortex2.h + vx_check_occupancy
#include <VX_types.h>

#include "ps_common.h"
#include "ps_phase.h"   // -P: persist_l2 sensitivity knob (refused unless -DPS_ALLOW_PERSIST_L2=1)
#include "ps_waymask.h"   // -W: mechanism A2 way masks for every launch (encoding, methodology §14.A)
#include "ps_q80.h"      // -q q80: host dequantize of the embedding row
#include "reused/gemv_f32/common.h"
#include "reused/rmsnorm/common.h"
#include "reused/rope/common.h"
#include "reused/silu_gate/common.h"
#include "reused/residual/common.h"
#include "reused/decode_attn_gqa/common.h"
#include "reused/linear/common.h"
#include "reused/prefill_attn/common.h"
#include "reused/gemv_q80/common.h"
#include "reused/rope/rope_table_ref.h"   // -R table: entry-by-entry host check of the device angle table
#include "reused/rope/common_kv8.h"         // -Q kv8: rope_kv_tab_q8 arguments
#include "reused/decode_attn_gqa/common_kv.h" // -Q kv8: attn_gqa_kv8 arguments and LMEM layout

#ifndef VX_CFG_DCACHE_SIZE
#define VX_CFG_DCACHE_SIZE 16384
#endif
#ifndef VX_CFG_DCACHE_NUM_WAYS
#define VX_CFG_DCACHE_NUM_WAYS 4
#endif
#ifndef VX_CFG_DCACHE_LINE_SIZE
#define VX_CFG_DCACHE_LINE_SIZE 64
#endif

#define RT_CHECK(_expr)                                                       \
  do {                                                                        \
    int _ret = (int)(_expr);                                                  \
    if (_ret != 0) {                                                          \
      std::printf("Error: '%s' returned %d!\nFAILED!\n", #_expr, _ret);       \
      std::exit(1);                                                           \
    }                                                                         \
  } while (false)

static void die(const std::string& msg) {
  std::printf("%s\nFAILED!\n", msg.c_str());
  std::exit(1);
}

static const double kTieRel = 1e-4;     // near-tie margin
static const double kLogitsRel = 1e-3;  // batched-prefill logits tolerance (norm-wise, see header)

// ------------------------------------------------------------------------------------------------- checkpoint
struct Config {
  int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
  bool shared;
  int head_size() const { return dim / n_heads; }
  int kv_dim() const { return dim * n_kv_heads / n_heads; }
  int kv_mul() const { return n_heads / n_kv_heads; }
};

// Float offsets into the weight blob (the checkpoint after its 28-byte header), in run.c's memory_map_weights order.
struct WOff {
  size_t emb, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final, wcls, total;
};

static WOff weight_offsets(const Config& c) {
  const size_t L = c.n_layers, d = c.dim, h = c.hidden_dim, kv = c.kv_dim(), V = c.vocab_size;
  WOff o{};
  size_t p = 0;
  o.emb = p; p += V * d;
  o.rms_att = p; p += L * d;
  o.wq = p; p += L * d * d;
  o.wk = p; p += L * d * kv;
  o.wv = p; p += L * d * kv;
  o.wo = p; p += L * d * d;
  o.rms_ffn = p; p += L * d;
  o.w1 = p; p += L * d * h;
  o.w2 = p; p += L * h * d;
  o.w3 = p; p += L * d * h;
  o.rms_final = p; p += d;
  p += (size_t)c.seq_len * c.head_size() / 2;  // freq_cis_real (unused since llama2.c computes RoPE on the fly)
  p += (size_t)c.seq_len * c.head_size() / 2;  // freq_cis_imag
  o.wcls = c.shared ? o.emb : p;
  if (!c.shared) p += V * d;
  o.total = p;
  return o;
}

static void load_checkpoint(const std::string& path, Config& c, std::vector<float>& w) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open checkpoint " + path);
  int32_t hdr[7];
  f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
  if (!f) die("short checkpoint header");
  c = {hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[5] > 0};
  c.vocab_size = std::abs(c.vocab_size);
  f.seekg(0, std::ios::end);
  const size_t bytes = (size_t)f.tellg() - sizeof hdr;
  const WOff o = weight_offsets(c);
  if (bytes != o.total * 4)
    die("checkpoint size mismatch: " + std::to_string(bytes) + " bytes of weights, expected " +
        std::to_string(o.total * 4));
  w.resize(o.total);
  f.seekg(sizeof hdr, std::ios::beg);
  f.read(reinterpret_cast<char*>(w.data()), (std::streamsize)bytes);
  if (!f) die("short checkpoint read");
}

// llama2.c version-2 (Q8_0) file, as runq.c read_checkpoint + memory_map_weights (byte offsets from the end of the 256-byte
// header; every tensor is `n` int8 values followed by n / GS float32 scales, per matrix one tensor per layer).
struct Q80File {
  int gs = 0;
  std::vector<uint8_t> blob;     // the file after the header: fp32 norms, then the Q8_0 tensors
  size_t rms_att = 0, rms_ffn = 0, rms_final = 0;   // byte offsets of the fp32 norms
  size_t tok = 0;                                    // token table tensor (also the classifier when shared)
  size_t wq = 0, wk = 0, wv = 0, wo = 0, w1 = 0, w2 = 0, w3 = 0;   // first layer's tensor of each matrix
  size_t wcls = 0;               // classifier tensor (== tok when shared)
  // bytes of one tensor of `n` values and the byte offset of its scales
  size_t tbytes(size_t n) const { return n + n / gs * 4; }
};

static void load_q80(const std::string& path, Config& c, Q80File& q) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open checkpoint " + path);
  uint8_t hdr[256];
  f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
  if (!f) die("short Q8_0 header");
  uint32_t magic;
  int32_t version, cfgi[7], gs;
  std::memcpy(&magic, hdr, 4);
  std::memcpy(&version, hdr + 4, 4);
  std::memcpy(cfgi, hdr + 8, 28);
  const uint8_t shared = hdr[36];
  std::memcpy(&gs, hdr + 37, 4);
  if (magic != 0x616b3432 || version != 2) die("not a llama2.c version-2 (Q8_0) checkpoint: " + path);
  c = {cfgi[0], cfgi[1], cfgi[2], cfgi[3], cfgi[4], cfgi[5], cfgi[6], shared != 0};
  c.vocab_size = std::abs(c.vocab_size);
  if (gs <= 0 || gs % 4 || c.dim % gs || c.hidden_dim % gs) die("unsupported Q8_0 group size " + std::to_string(gs));
  q.gs = gs;
  f.seekg(0, std::ios::end);
  const size_t total = (size_t)f.tellg() - sizeof hdr;
  const size_t L = c.n_layers, d = c.dim, h = c.hidden_dim, kv = c.kv_dim(), V = c.vocab_size;
  size_t p = 0;
  q.rms_att = p; p += L * d * 4;
  q.rms_ffn = p; p += L * d * 4;
  q.rms_final = p; p += d * 4;
  q.tok = p; p += q.tbytes(V * d);
  q.wq = p; p += L * q.tbytes(d * d);
  q.wk = p; p += L * q.tbytes(d * kv);
  q.wv = p; p += L * q.tbytes(d * kv);
  q.wo = p; p += L * q.tbytes(d * d);
  q.w1 = p; p += L * q.tbytes(d * h);
  q.w2 = p; p += L * q.tbytes(h * d);
  q.w3 = p; p += L * q.tbytes(d * h);
  q.wcls = c.shared ? q.tok : p;
  if (!c.shared) p += q.tbytes(V * d);
  if (p != total) die("Q8_0 checkpoint size mismatch: " + std::to_string(total) + " bytes of weights, expected " + std::to_string(p));
  q.blob.resize(total);
  f.seekg(sizeof hdr, std::ios::beg);
  f.read(reinterpret_cast<char*>(q.blob.data()), (std::streamsize)total);
  if (!f) die("short Q8_0 checkpoint read");
}

static void load_golden(const std::string& path, std::vector<int>& seq, std::vector<int>& forced) {
  std::ifstream f(path);
  if (!f) die("cannot open golden token file " + path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    int pos, tok, nxt, frc;
    if (!(is >> pos >> tok >> nxt >> frc)) die("bad golden line: " + line);
    if (pos != (int)forced.size()) die("golden positions not consecutive at " + line);
    if (seq.empty()) seq.push_back(tok);
    else if (seq.back() != tok) die("golden token chain broken at " + line);
    seq.push_back(nxt);
    forced.push_back(frc);
  }
  if (forced.empty()) die("empty golden token file");
}

// ------------------------------------------------------------------------------------------------- device
struct Buf {
  vx_buffer_h h = nullptr;
  uint64_t addr = 0, bytes = 0;
};

struct Counters {
  uint64_t cycles = 0, instrs = 0, mem_r = 0, mem_w = 0, mem_lt = 0, idle = 0, scrb = 0;
};

struct Dev {
  vx_device_h dev = nullptr;
  vx_queue_h q = nullptr;
  vx_module_h mod = nullptr;
  uint64_t nt = 0, nw = 0, ncores = 0, lmem_cap = 0;
  int mpm_class = 0;
  std::vector<Buf> bufs;
  std::vector<vx_kernel_h> kernels;
  Counters last;
  uint64_t launches = 0;

  void open(const std::string& kernel_file) {
    RT_CHECK(vx_device_open(0, &dev));
    vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
    RT_CHECK(vx_queue_create(dev, &qi, &q));
    RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
    RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
    RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &ncores));
    RT_CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
    RT_CHECK(vx_module_load_file(dev, kernel_file.c_str(), &mod));
    const char* s = std::getenv("VORTEX_PROFILING");
    mpm_class = s ? std::atoi(s) : 0;
    last = read_counters();
  }
  vx_kernel_h kernel(const char* name) {
    vx_kernel_h k = nullptr;
    RT_CHECK(vx_module_get_kernel(mod, name, &k));
    kernels.push_back(k);
    return k;
  }
  Buf alloc(uint64_t bytes, const void* init) {
    Buf b;
    b.bytes = bytes;
    RT_CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &b.h));
    RT_CHECK(vx_buffer_address(b.h, &b.addr));
    if (init) RT_CHECK(vx_enqueue_write(q, b.h, 0, init, bytes, 0, nullptr, nullptr));
    bufs.push_back(b);
    return b;
  }
  void finish() { RT_CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE)); }
  void copy(const Buf& dst, uint64_t doff, const Buf& src, uint64_t soff, uint64_t bytes) {
    RT_CHECK(vx_enqueue_copy(q, dst.h, doff, src.h, soff, bytes, 0, nullptr, nullptr));
  }
  void read(void* host, const Buf& b, uint64_t off, uint64_t bytes) {
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_read(q, host, b.h, off, bytes, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }
  uint64_t mpm(uint32_t addr, uint32_t core) {
    uint64_t v = 0;
    RT_CHECK(vx_device_mpm_query(dev, (uint32_t)mpm_class, addr, core, &v));
    return v;
  }
  Counters read_counters() {
    Counters c;
    for (uint32_t i = 0; i < ncores; ++i) c.cycles = std::max(c.cycles, mpm(VX_CSR_MCYCLE, i));
    c.instrs = mpm(VX_CSR_MINSTRET, 0xffffffffu);
    if (mpm_class == VX_DCR_MPM_CLASS_MEM) {
      c.mem_r = mpm(VX_CSR_MPM_MEM_READS, 0);
      c.mem_w = mpm(VX_CSR_MPM_MEM_WRITES, 0);
      c.mem_lt = mpm(VX_CSR_MPM_MEM_LT, 0);
    } else if (mpm_class == VX_DCR_MPM_CLASS_CORE) {
      c.idle = mpm(VX_CSR_MPM_SCHED_IDLE, 0xffffffffu);
      c.scrb = mpm(VX_CSR_MPM_STALL_SCRB, 0xffffffffu);
    }
    return c;
  }
  // One launch, waited for; returns the counter deltas of this launch and prints its PSL line.
  Counters launch(vx_kernel_h k, const char* kname, const void* arg, size_t arg_size, uint32_t grid,
                  uint32_t block, uint32_t lmem, int pos, int layer, const char* op) {
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.args_host = arg;
    li.args_size = arg_size;
    li.ndim = 1;
    li.grid_dim[0] = grid;
    li.block_dim[0] = block;
    li.lmem_size = lmem;
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    ++launches;
    Counters now = read_counters(), d;
    d.cycles = now.cycles - last.cycles;
    d.instrs = now.instrs - last.instrs;
    d.mem_r = now.mem_r - last.mem_r;
    d.mem_w = now.mem_w - last.mem_w;
    d.mem_lt = now.mem_lt - last.mem_lt;
    d.idle = now.idle - last.idle;
    d.scrb = now.scrb - last.scrb;
    last = now;
    std::printf("PSL: pos=%d layer=%d op=%s kernel=%s grid=%u block=%u lmem=%u cycles=%llu instrs=%llu mem_r=%llu "
                "mem_w=%llu mem_lt=%llu idle=%llu scrb=%llu\n",
                pos, layer, op, kname, grid, block, lmem, (unsigned long long)d.cycles, (unsigned long long)d.instrs,
                (unsigned long long)d.mem_r, (unsigned long long)d.mem_w, (unsigned long long)d.mem_lt,
                (unsigned long long)d.idle, (unsigned long long)d.scrb);
    return d;
  }
  // 1-D launch of `global` work items with the runtime's maximum-occupancy geometry (as the apps)
  Counters launch1d(vx_kernel_h k, const char* kname, const void* arg, size_t arg_size, uint32_t global, int pos,
                    int layer, const char* op) {
    uint32_t grid[1], block[1];
    RT_CHECK(vx_device_max_occupancy_grid(dev, 1, &global, grid, block));
    if (block[0] % nt != 0) die("block not a multiple of NT");
    return launch(k, kname, arg, arg_size, grid[0], block[0], 0, pos, layer, op);
  }
  void release() {
    for (auto& b : bufs) if (b.h) vx_buffer_release(b.h);
    for (auto k : kernels) vx_kernel_release(k);
    if (mod) vx_module_release(mod);
    if (q) vx_queue_release(q);
    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);
  }
};

// ------------------------------------------------------------------------------------------------- GEMV variant rule
// Max number of the NT rows of one warp that share an L1 set, over every warp-aligned first row and every 16-byte
// offset within a line (header comment "GEMV variant").
static uint32_t rows_set_pressure(uint64_t stride_bytes, uint32_t nt) {
  const uint32_t line = VX_CFG_DCACHE_LINE_SIZE;
  const uint32_t sets = VX_CFG_DCACHE_SIZE / (VX_CFG_DCACHE_LINE_SIZE * VX_CFG_DCACHE_NUM_WAYS);
  const uint64_t period = (uint64_t)line * sets;  // the set pattern repeats every `period` bytes of r * stride
  uint32_t worst = 0;
  std::vector<uint32_t> cnt(sets);
  for (uint64_t r0 = 0; r0 < period; r0 += nt) {
    for (uint32_t a = 0; a < line; a += 16) {
      std::fill(cnt.begin(), cnt.end(), 0u);
      for (uint32_t j = 0; j < nt; ++j) {
        const uint64_t byte = a + (r0 + j) * stride_bytes;
        worst = std::max(worst, ++cnt[(byte / line) % sets]);
      }
    }
  }
  return worst;
}

struct Gemv {
  const char* name;
  uint32_t M, N;
  bool rows;
};

// ------------------------------------------------------------------------------------------------- helpers
static void top2(const std::vector<float>& l, int& am, double& l1, double& l2) {
  am = 0;
  for (int i = 1; i < (int)l.size(); ++i)
    if (l[i] > l[am]) am = i;  // first index of the maximum (run.c argmax)
  l1 = l[am];
  l2 = -INFINITY;
  for (int i = 0; i < (int)l.size(); ++i)
    if (i != am) l2 = std::max(l2, (double)l[i]);
}

static double rel_margin(double a, double b) {
  if (a == 0.0) return b == 0.0 ? 0.0 : INFINITY;
  return std::fabs(a - b) / std::fabs(a);
}

// prefill attention tiles: the host rule (kernels/prefill_attn/main.cpp "LMEM and tiles"), lmem variant
static uint32_t max_tk(uint32_t G, uint32_t D, uint32_t Qt, uint64_t cap) {
  for (uint32_t tk = 32; tk >= 1; --tk)
    if (ps_pattn_lmem(G, D, Qt, tk, 1).total * 4ull <= cap) return tk;
  return 0;
}

// ------------------------------------------------------------------------------------------------- main
int main(int argc, char** argv) {
  std::string model, golden, mode = "tf", variant = "auto", ref_path, out_path, kernel_file = "kernel.vxbin", qmode = "f32",
              rmode = "perlaunch", emode = "musl", kmode = "fp32";
  int npos = 32;
  uint32_t b_cap = 1;
  bool persist = false;
  bool have_wm = false;   // PhaseShift: -W way masks (absent = no DCR write, the old host path)
  uint32_t wm_l1 = 0, wm_l2 = 0;
  int c;
  while ((c = getopt(argc, argv, "w:T:m:n:v:c:r:o:k:q:R:E:Q:PW:h")) != -1) {
    switch (c) {
      case 'w': model = optarg; break;
      case 'T': golden = optarg; break;
      case 'm': mode = optarg; break;
      case 'n': npos = std::atoi(optarg); break;
      case 'v': variant = optarg; break;
      case 'c': b_cap = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'r': ref_path = optarg; break;
      case 'o': out_path = optarg; break;
      case 'k': kernel_file = optarg; break;
      case 'q': qmode = optarg; break;
      case 'R': rmode = optarg; break;
      case 'E': emode = optarg; break;
      case 'Q': kmode = optarg; break;
      case 'P': persist = true; break;
      case 'W':   // PhaseShift: [dcr_phase] way masks "<l1word>:<l2word>" for every launch
        if (!ps::parse_waymask_arg(optarg, &wm_l1, &wm_l2)) die(std::string("bad -W ") + optarg);
        have_wm = true;
        break;
      default:
        std::printf("Usage: %s -w model.bin -T golden_tokens.txt [-m tf|free|prefill] [-n positions] "
                    "[-v auto|rows|lanes] [-c b_cap] [-r ref_logits.f32] [-o logits.f32] [-k kernel.vxbin] [-q f32|q80] [-R perlaunch|table] [-E musl|poly|mufu] [-Q fp32|kv8] [-P] [-W l1:l2]\n"
                    "  -P  persist_l2 sensitivity knob (ps tree, host built with -DPS_ALLOW_PERSIST_L2=1)\n"
                    "  -W  [dcr_phase] way masks <l1word>:<l2word> for every launch (cache way-mask mechanism)\n",
                    argv[0]);
        return c == 'h' ? 0 : 1;
    }
  }
  if (model.empty() || golden.empty()) die("-w and -T are required");
  if (mode != "tf" && mode != "free" && mode != "prefill") die("bad -m " + mode);
  if (variant != "auto" && variant != "rows" && variant != "lanes") die("bad -v " + variant);
  if (qmode != "f32" && qmode != "q80") die("bad -q " + qmode);
  if (rmode != "perlaunch" && rmode != "table") die("bad -R " + rmode);
  if (emode != "musl" && emode != "poly" && emode != "mufu") die("bad -E " + emode);
  if (kmode != "fp32" && kmode != "kv8") die("bad -Q " + kmode);
  const bool kv8 = kmode == "kv8";
  if (kv8) {
    if (qmode != "q80") die("-Q kv8 is the P8K8 track: it needs -q q80");
    if (mode == "prefill") die("-Q kv8 supports the decode modes tf and free only");
    if (rmode != "table") { std::printf("note: -Q kv8 implies -R table (rope_kv_tab_q8 loads the angle table)\n"); rmode = "table"; }
  }
  const bool q8 = qmode == "q80";
  const bool tabm = rmode == "table";
  if (q8 && mode == "prefill") die("-q q80 supports the decode modes tf and free only");
  if (b_cap < 1) die("-c must be >= 1");
  if (mode == "prefill" && ref_path.empty()) die("prefill mode is validated by its logits: -r is required");

  Config cfg;
  std::vector<float> W;
  Q80File qf;
  if (q8) load_q80(model, cfg, qf);
  else load_checkpoint(model, cfg, W);
  std::vector<int> seq, forced;
  load_golden(golden, seq, forced);
  const int dim = cfg.dim, hid = cfg.hidden_dim, L = cfg.n_layers, kvd = cfg.kv_dim(), hs = cfg.head_size(),
            V = cfg.vocab_size, S_max = cfg.seq_len, kv_mul = cfg.kv_mul();
  if (npos < 1 || npos > S_max || npos > (int)forced.size()) die("bad -n (positions)");
  int P = 1;  // prompt length = 1 + forced lines (forced[i] = 1: seq[i+1] is a prompt token)
  while (P - 1 < (int)forced.size() && forced[P - 1]) ++P;
  if (mode == "prefill" && P > npos) die("-n must cover the prompt in prefill mode");
  if (kv8 && hs % 4) die("-Q kv8 needs head_size % 4 == 0");
  if (kv_mul != 1 && kv_mul != 2 && kv_mul != 4 && kv_mul != 8)
    die("decode attention supports G = kv_mul in {1, 2, 4, 8}; this model has " + std::to_string(kv_mul));
  const WOff wo = weight_offsets(cfg);   // fp32 layout; unused with -q q80 (qf holds the Q8_0 offsets)
  std::vector<float> ref;
  int n_ref = 0;
  if (!ref_path.empty()) {
    std::ifstream f(ref_path, std::ios::binary | std::ios::ate);
    if (!f) die("cannot open " + ref_path);
    const size_t b = (size_t)f.tellg();
    if (b % ((size_t)V * 4)) die("reference logits size is not a multiple of vocab_size floats");
    n_ref = (int)(b / ((size_t)V * 4));
    ref.resize(b / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(ref.data()), (std::streamsize)b);
  }
  std::printf("ps_llama: model=%s dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d shared=%d "
              "head_size=%d kv_dim=%d kv_mul=%d mode=%s positions=%d prompt=%d b_cap=%u variant=%s ref_positions=%d quant=%s gs=%d\n",
              model.c_str(), dim, hid, L, cfg.n_heads, cfg.n_kv_heads, V, S_max, (int)cfg.shared, hs, kvd, kv_mul,
              mode.c_str(), npos, P, b_cap, variant.c_str(), n_ref, qmode.c_str(), q8 ? qf.gs : 0);

  Dev d;
  d.open(kernel_file);
  if (persist && !have_wm) {   // PhaseShift: [dcr_phase] persist_l2 for every launch of this run (labelled sensitivity study)
    ps::PhaseCfg pc;
    pc.flags = ps::phase_dcr::FLAG_PERSIST_L2;
    if (ps::phase_enqueue(d.q, pc) != 0) die("-P: persist_l2 refused or DCR write failed");
    std::printf("PSPERSIST-HOST: persist_l2=1 (sensitivity study)\n");
  } else if (have_wm) {   // PhaseShift: way masks, one block for the whole run (masks change only here = before
                          // launch 0, so no mid-run mask change; with -P SimX rejects the block by design)
    ps::PhaseCfg pc;
    if (persist) pc.flags = ps::phase_dcr::FLAG_PERSIST_L2;
    ps::WayMaskCtl wmc;
    if (wmc.set(pc, wm_l1, wm_l2) != 0) die("-W: mask change refused");
    ps::waymask_print(wmc);
    if (ps::phase_enqueue(d.q, pc) != 0) die("-W: DCR write failed");
  }
  const uint32_t nt = (uint32_t)d.nt, nth = (uint32_t)(d.nt * d.nw);
  std::printf("device: NT=%u NW=%u cores=%llu lmem=%llu L1=%u B %u-way %u B lines mpm_class=%d\n", nt,
              (unsigned)d.nw, (unsigned long long)d.ncores, (unsigned long long)d.lmem_cap, VX_CFG_DCACHE_SIZE,
              VX_CFG_DCACHE_NUM_WAYS, VX_CFG_DCACHE_LINE_SIZE, d.mpm_class);

  vx_kernel_h k_rows = d.kernel("gemv_rows"), k_lanes = d.kernel("gemv_lanes"), k_rms = d.kernel("rmsnorm"),
              k_rope = d.kernel("rope_kv"), k_silu = d.kernel("silu_gate"), k_res = d.kernel("residual"),
              k_attn = d.kernel("attn_gqa"), k_gemm = d.kernel("gemm_small"), k_pattn = d.kernel("attn_prefill_lmem");
  vx_kernel_h kq_rows = nullptr, kq_lanes = nullptr, k_quant = nullptr, k_rtab = nullptr, k_rkvt = nullptr, k_rkv8 = nullptr, k_attn8 = nullptr;
  if (kv8) {   // (translation units 11, 12)
    k_rkv8 = d.kernel("rope_kv_tab_q8");
    k_attn8 = d.kernel("attn_gqa_kv8");
  }
  if (tabm) {   // (translation unit 10)
    k_rtab = d.kernel("rope_table");
    k_rkvt = d.kernel("rope_kv_tab");
  }
  const char* silu_name = emode == "poly" ? "silu_gate_expf_poly" : emode == "mufu" ? "silu_gate_expf_mufu" : "silu_gate";
  if (emode != "musl") k_silu = d.kernel(silu_name);
  if (emode == "mufu") {
    uint64_t isa = 0;
    RT_CHECK(vx_device_query(d.dev, VX_CAPS_ISA_FLAGS, &isa));
    if (!(isa & VX_ISA_EXT_MUFU)) die("-E mufu needs a device built with -DVX_CFG_EXT_MUFU_ENABLE (ISA flag VX_ISA_EXT_MUFU)");
  }
  std::printf("PSEPK: kv=%s kv_bytes_per_token_layer_fp32=%d kv_bytes_per_token_layer_int8=%d scale_bytes_per_token_layer=%d\n", kmode.c_str(), 2 * cfg.kv_dim() * 4,
              2 * cfg.kv_dim(), 2 * cfg.n_kv_heads * 4);
  std::printf("PSEP1: rope=%s exp=%s numerics_changing=%d silu_kernel=%s\n", rmode.c_str(), emode.c_str(), (int)(emode != "musl"),
              silu_name);
  if (q8) {
    kq_rows = d.kernel("gemv_q80_rows");
    kq_lanes = d.kernel("gemv_q80_lanes");
    k_quant = d.kernel("quantize_q80");
  }

  // ---- GEMV variants per matrix shape (rule in the header)
  // (q80: the weight rows are int8, so the row stride is N bytes, not 4 N)
  auto choose = [&](const char* name, uint32_t M, uint32_t N) {
    const uint32_t stride = q8 ? N : N * 4;
    const uint32_t pressure = rows_set_pressure((uint64_t)stride, nt);
    const bool rows = variant == "rows" || (variant == "auto" && pressure <= VX_CFG_DCACHE_NUM_WAYS);
    std::printf("PSGEMV: matrix=%s M=%u K=%u stride_bytes=%u stride_lines=%.4g rows_set_pressure=%u ways=%u "
                "variant=%s%s\n", name, M, N, stride, stride * 1.0 / VX_CFG_DCACHE_LINE_SIZE, pressure,
                VX_CFG_DCACHE_NUM_WAYS, rows ? "rows" : "lanes", variant == "auto" ? "" : " (forced by -v)");
    return Gemv{name, M, N, rows};
  };
  const Gemv g_q = choose("wq", dim, dim), g_k = choose("wk", kvd, dim), g_v = choose("wv", kvd, dim),
             g_o = choose("wo", dim, dim), g_1 = choose("w1", hid, dim), g_3 = choose("w3", hid, dim),
             g_2 = choose("w2", dim, hid), g_cls = choose("wcls", V, dim);

  // ---- device memory: weights (one blob), KV cache (poisoned with NaN: a read of a slot no rope_kv wrote shows up
  // as NaN logits), activations (rows = 1 for decode, P for the batched prefill)
  const uint32_t R = (mode == "prefill") ? (uint32_t)P : 1u;
  Buf bw = q8 ? d.alloc((uint64_t)qf.blob.size(), qf.blob.data()) : d.alloc((uint64_t)W.size() * 4, W.data());
  if (!q8)
    for (size_t off : {wo.emb, wo.rms_att, wo.wq, wo.wk, wo.wv, wo.wo, wo.rms_ffn, wo.w1, wo.w2, wo.w3, wo.rms_final,
                       wo.wcls})
      if ((bw.addr + off * 4) % 16) std::printf("note: a weight array is not 16-byte aligned (scalar kernel path)\n");
  const size_t kv_n = (size_t)L * b_cap * S_max * kvd;
  const size_t sc_n = (size_t)L * b_cap * cfg.n_kv_heads * S_max;   // kv8: scales [layer][b][kv_head][pos]
  std::vector<float> nanv(kv8 ? sc_n : kv_n, std::nanf(""));   // must outlive the queued writes (the queue reads them at d.finish())
  std::vector<uint8_t> poison8(kv8 ? kv_n : 0, 0x55);
  Buf bkc, bvc, bks, bvs;
  if (kv8) {
    bkc = d.alloc(kv_n, poison8.data()); bvc = d.alloc(kv_n, poison8.data());
    bks = d.alloc(sc_n * 4, nanv.data()); bvs = d.alloc(sc_n * 4, nanv.data());
  } else {
    bkc = d.alloc(kv_n * 4, nanv.data()); bvc = d.alloc(kv_n * 4, nanv.data());
  }
  auto act = [&](uint64_t floats) { return d.alloc(floats * 4, nullptr); };
  Buf bx = act((uint64_t)R * dim), bxb = act((uint64_t)R * dim), bq = act((uint64_t)R * dim),
      bk = act((uint64_t)R * kvd), bv = act((uint64_t)R * kvd), bao = act((uint64_t)R * dim),
      bxo = act((uint64_t)R * dim), bh1 = act((uint64_t)R * hid), bh3 = act((uint64_t)R * hid),
      blog = act((uint64_t)V), bpos = d.alloc((uint64_t)R * 4, nullptr);
  // -R table: the angle table, seq_len x head_size/2 x {cos, sin} (poisoned with NaN: an unwritten entry shows in the check)
  const size_t tab_n = (size_t)S_max * (hs / 2) * 2;
  Buf btab;
  std::vector<float> tpois;   // must outlive the queued write (the queue reads it at d.finish(), not at enqueue)
  if (tabm) {
    tpois.assign(tab_n, std::nanf(""));
    btab = d.alloc(tab_n * 4, tpois.data());
  }
  // q80: quantized activations (runq.c xq: dim values, hq: hidden_dim values) and their group scales
  const uint32_t GSq = q8 ? (uint32_t)qf.gs : 1u;
  Buf bxq, bxs, bhq, bhs;
  if (q8) {
    bxq = d.alloc((uint64_t)dim, nullptr);  bxs = d.alloc((uint64_t)dim / GSq * 4, nullptr);
    bhq = d.alloc((uint64_t)hid, nullptr);  bhs = d.alloc((uint64_t)hid / GSq * 4, nullptr);
  }
  d.finish();
  const uint64_t W0 = bw.addr;
  auto wa = [&](size_t off) { return W0 + off * 4; };
  const int launches_per_token = q8 ? 18 * L + 3 : 14 * L + 2;
  std::printf("memory: weights %.2f MiB, kv cache 2 x %.2f MiB (b_cap=%u%s), launches per decode token = %d\n",
              (q8 ? qf.blob.size() : W.size() * 4) / 1048576.0, (kv8 ? kv_n * 1.0 + sc_n * 4.0 : kv_n * 4.0) / 1048576.0, b_cap, kv8 ? ", int8 + fp32 scales" : "", launches_per_token);
  if (q8)
    std::printf("PSQ80: gs=%d weight_bytes_per_token=%.0f bytes_per_weight=%.4f\n", qf.gs,
                (double)(L * (qf.tbytes((size_t)dim * dim) * 2 + qf.tbytes((size_t)dim * kvd) * 2 + qf.tbytes((size_t)dim * hid) * 3) +
                         qf.tbytes((size_t)V * dim)),
                (double)qf.tbytes(1024) / 1024.0);

  // ---- op launchers
  auto gemv = [&](const Gemv& g, uint64_t Wa, const Buf& x, uint64_t xoff, const Buf& y, int pos, int layer) {
    ps_gemv_arg_t a = {};
    a.W_addr = Wa; a.x_addr = x.addr + xoff; a.y_addr = y.addr; a.M = g.M; a.N = g.N;
    return g.rows ? d.launch1d(k_rows, "gemv_rows", &a, sizeof a, g.M, pos, layer, g.name)
                  : d.launch1d(k_lanes, "gemv_lanes", &a, sizeof a, g.M * nt, pos, layer, g.name);
  };
  auto gemm = [&](const char* name, uint32_t N, uint32_t K, uint64_t Wa, const Buf& x, const Buf& y, uint32_t M,
                  int pos, int layer) {
    ps_gemm_small_arg_t a = {};
    a.W_addr = Wa; a.X_addr = x.addr; a.Y_addr = y.addr; a.M = M; a.N = N; a.K = K;
    return d.launch1d(k_gemm, "gemm_small", &a, sizeof a, N, pos, layer, name);
  };
  // q80: byte offset of layer l's tensor of a matrix (`first` = qf.wq etc.; one tensor = q values then scales)
  auto qt = [&](size_t first, int l, size_t n) { return W0 + first + (size_t)l * qf.tbytes(n); };
  auto gemvq = [&](const Gemv& g, uint64_t tensor, const Buf& xq, const Buf& xs, const Buf& y, int pos, int layer) {
    ps_gemv_q80_arg_t a = {};
    a.Wq_addr = tensor; a.Ws_addr = tensor + (uint64_t)g.M * g.N; a.xq_addr = xq.addr; a.xs_addr = xs.addr;
    a.y_addr = y.addr; a.M = g.M; a.N = g.N; a.GS = GSq; a.B = 1;
    return g.rows ? d.launch1d(kq_rows, "gemv_q80_rows", &a, sizeof a, g.M, pos, layer, g.name)
                  : d.launch1d(kq_lanes, "gemv_q80_lanes", &a, sizeof a, g.M * nt, pos, layer, g.name);
  };
  auto quant = [&](const Buf& x, const Buf& q, const Buf& s, uint32_t n, int pos, int layer, const char* op) {
    ps_quant_q80_arg_t a = {};
    a.x_addr = x.addr; a.q_addr = q.addr; a.s_addr = s.addr; a.B = 1; a.n = n; a.GS = GSq;
    return d.launch1d(k_quant, "quantize_q80", &a, sizeof a, n / GSq, pos, layer, op);
  };
  auto rmsnorm = [&](const Buf& x, uint64_t xoff, uint64_t wgt, const Buf& out, uint64_t ooff, uint32_t B, int pos,
                     int layer, const char* op) {
    ps_rmsnorm_arg_t a = {};
    a.x_addr = x.addr + xoff; a.w_addr = wgt; a.out_addr = out.addr + ooff; a.B = B; a.n = dim; a.eps = 1e-5f;
    return d.launch1d(k_rms, "rmsnorm", &a, sizeof a, B * nt, pos, layer, op);  // one warp per row
  };
  auto residual = [&](uint32_t B, int pos, int layer, const char* op) {  // x += xo (run.c x[i] += xb[i])
    ps_residual_arg_t a = {};
    a.x_addr = bx.addr; a.branch_addr = bxo.addr; a.out_addr = bx.addr; a.n = B * dim;
    return d.launch1d(k_res, "residual", &a, sizeof a, B * dim, pos, layer, op);
  };
  auto silu = [&](uint32_t B, int pos, int layer) {  // hb = silu(hb) * hb2, in place as run.c
    ps_silu_gate_arg_t a = {};
    a.hb_addr = bh1.addr; a.hb2_addr = bh3.addr; a.out_addr = bh1.addr; a.n = B * hid;
    return d.launch1d(k_silu, silu_name, &a, sizeof a, B * hid, pos, layer, "silu_gate");
  };
  const uint64_t layer_kv_bytes = (uint64_t)b_cap * S_max * kvd * (kv8 ? 1 : 4);  // one layer of the cache, all b_cap rows
  const uint64_t layer_sc_bytes = (uint64_t)b_cap * cfg.n_kv_heads * S_max * 4;     // kv8: one layer of the scales
  // decode: one row (sequence b = 0) at position pos; prefill: B = P rows of sequence 0 at positions 0..P-1. The
  // prefill launch writes every row into sequence 0's slots by passing the layer's sequence-0 base with seq_len = 0
  // and layer = 0, so the kernel's slot ((layer*b_cap + b)*seq_len + pos)*kv_dim reduces to pos*kv_dim for every row
  // b (host-side use of the unchanged kernel; methodology §14.5).
  auto rope = [&](uint32_t B, int layer, int pos) {
    ps_rope_arg_t a = {};
    a.q_addr = bq.addr; a.k_addr = bk.addr; a.v_addr = bv.addr; a.pos_addr = bpos.addr; a.q_out_addr = bq.addr;
    a.B = B; a.dim = dim; a.kv_dim = kvd; a.head_size = hs; a.theta = 10000.0f;
    if (B == 1) {
      a.kc_addr = bkc.addr; a.vc_addr = bvc.addr; a.b_cap = b_cap; a.seq_len = S_max; a.layer = layer;
    } else {
      a.kc_addr = bkc.addr + layer * layer_kv_bytes; a.vc_addr = bvc.addr + layer * layer_kv_bytes;
      a.b_cap = 1; a.seq_len = 0; a.layer = 0;
    }
    if (kv8) {   // B == 1 only (decode); same arguments as rope_kv_tab, the caches are int8 + scales
      ps_rope_q8_arg_t qa = {};
      qa.t.a = a; qa.t.tab_addr = btab.addr; qa.ks_addr = bks.addr; qa.vs_addr = bvs.addr; qa.H_kv = (uint32_t)cfg.n_kv_heads;
      return d.launch1d(k_rkv8, "rope_kv_tab_q8", &qa, sizeof qa, B * (dim / 2) + 2 * B * (uint32_t)cfg.n_kv_heads, pos, layer, "rope_kv");
    }
    if (tabm) {   // angles from the table (same arguments plus the table address)
      ps_rope_tab_arg_t ta = {};
      ta.a = a; ta.tab_addr = btab.addr;
      return d.launch1d(k_rkvt, "rope_kv_tab", &ta, sizeof ta, B * (dim / 2), pos, layer, "rope_kv");
    }
    return d.launch1d(k_rope, "rope_kv", &a, sizeof a, B * (dim / 2), pos, layer, "rope_kv");
  };
  const uint32_t G = kv_mul, Tt = 32;
  const uint32_t attn_lmem = (kv8 ? ps_attn_kv_lmem(G, hs, Tt, 1, 0, 1).total : ps_attn_lmem(G, hs, Tt, 1, 0).total) * 4;
  if (attn_lmem > d.lmem_cap) die("decode attention LMEM footprint exceeds the core LMEM");
  RT_CHECK(vx_check_occupancy(d.dev, nth, attn_lmem));
  auto attn = [&](int layer, int pos) {
    ps_attn_arg_t a = {};
    a.q_addr = bq.addr; a.k_addr = bkc.addr + layer * layer_kv_bytes; a.v_addr = bvc.addr + layer * layer_kv_bytes;
    a.o_addr = bao.addr; a.B = 1; a.H_q = cfg.n_heads; a.H_kv = cfg.n_kv_heads; a.D = hs; a.T = pos + 1;
    a.Tmax = S_max; a.kv_mul = kv_mul; a.G = G; a.Tt = Tt; a.depth = 1; a.scale = 1.0f / sqrtf((float)hs);
    if (kv8) {
      ps_attn_kv_arg_t ka = {};
      ka.a = a; ka.ks_addr = bks.addr + layer * layer_sc_bytes; ka.vs_addr = bvs.addr + layer * layer_sc_bytes; ka.esz = 1;
      return d.launch(k_attn8, "attn_gqa_kv8", &ka, sizeof ka, cfg.n_heads / G, nth, attn_lmem, pos, layer, "attn");
    }
    return d.launch(k_attn, "attn_gqa", &a, sizeof a, cfg.n_heads / G, nth, attn_lmem, pos, layer, "attn");
  };
  // prefill attention tiles (rule, lmem variant)
  uint32_t Qt = 0, Tk = 0;
  if (mode == "prefill") {
    for (uint32_t cand = 64; cand >= 1; cand >>= 1) {
      if (cand > (uint32_t)P && cand > 1) continue;
      if (G * cand > 128) continue;
      const uint32_t tk = max_tk(G, hs, cand, d.lmem_cap);
      if (tk >= 8) { Qt = cand; Tk = tk; break; }
    }
    if (Qt == 0) { Qt = 1; Tk = max_tk(G, hs, 1, d.lmem_cap); }
    while (Qt > 1 && (uint64_t)cfg.n_kv_heads * ((P + Qt - 1) / Qt) < d.ncores) { Qt >>= 1; Tk = max_tk(G, hs, Qt, d.lmem_cap); }
    if (Tk == 0) die("no prefill attention tile fits the LMEM");
  }
  const uint32_t pattn_lmem = (mode == "prefill") ? ps_pattn_lmem(G, hs, Qt, Tk, 1).total * 4 : 0;
  auto pattn = [&](int layer, int pos) {
    ps_pattn_arg_t a = {};
    a.q_addr = bq.addr; a.k_addr = bkc.addr + layer * layer_kv_bytes; a.v_addr = bvc.addr + layer * layer_kv_bytes;
    a.o_addr = bao.addr; a.B = 1; a.H_q = cfg.n_heads; a.H_kv = cfg.n_kv_heads; a.D = hs; a.S = P; a.s0 = 0;
    a.Tmax = S_max; a.G = G; a.Qt = Qt; a.Tk = Tk; a.depth = 1; a.nqt = (P + Qt - 1) / Qt;
    a.scale = 1.0f / sqrtf((float)hs);
    RT_CHECK(vx_check_occupancy(d.dev, nth, pattn_lmem));
    return d.launch(k_pattn, "attn_prefill_lmem", &a, sizeof a, cfg.n_kv_heads * a.nqt, nth, pattn_lmem, pos, layer,
                    "attn");
  };
  if (mode == "prefill")
    std::printf("prefill: S=%d Qt=%u Tk=%u lmem=%u attention grid=%u\n", P, Qt, Tk, pattn_lmem,
                cfg.n_kv_heads * ((P + Qt - 1) / Qt));

  // ---- one forward pass over R rows (R = 1: position pos; R = P: positions 0..P-1); leaves logits in blog
  auto forward = [&](uint32_t Rr, int pos_tag) {
    Counters tot;
    auto add = [&](const Counters& x) { tot.cycles += x.cycles; tot.instrs += x.instrs; };
    for (int l = 0; l < L; ++l) {
      add(rmsnorm(bx, 0, wa(wo.rms_att + (size_t)l * dim), bxb, 0, Rr, pos_tag, l, "rmsnorm_att"));
      if (Rr == 1) {
        add(gemv(g_q, wa(wo.wq + (size_t)l * dim * dim), bxb, 0, bq, pos_tag, l));
        add(gemv(g_k, wa(wo.wk + (size_t)l * dim * kvd), bxb, 0, bk, pos_tag, l));
        add(gemv(g_v, wa(wo.wv + (size_t)l * dim * kvd), bxb, 0, bv, pos_tag, l));
        add(rope(1, l, pos_tag));
        add(attn(l, pos_tag));
        add(gemv(g_o, wa(wo.wo + (size_t)l * dim * dim), bao, 0, bxo, pos_tag, l));
      } else {
        add(gemm("wq", dim, dim, wa(wo.wq + (size_t)l * dim * dim), bxb, bq, Rr, pos_tag, l));
        add(gemm("wk", kvd, dim, wa(wo.wk + (size_t)l * dim * kvd), bxb, bk, Rr, pos_tag, l));
        add(gemm("wv", kvd, dim, wa(wo.wv + (size_t)l * dim * kvd), bxb, bv, Rr, pos_tag, l));
        add(rope(Rr, l, pos_tag));
        add(pattn(l, pos_tag));
        add(gemm("wo", dim, dim, wa(wo.wo + (size_t)l * dim * dim), bao, bxo, Rr, pos_tag, l));
      }
      add(residual(Rr, pos_tag, l, "residual_att"));
      add(rmsnorm(bx, 0, wa(wo.rms_ffn + (size_t)l * dim), bxb, 0, Rr, pos_tag, l, "rmsnorm_ffn"));
      if (Rr == 1) {
        add(gemv(g_1, wa(wo.w1 + (size_t)l * dim * hid), bxb, 0, bh1, pos_tag, l));
        add(gemv(g_3, wa(wo.w3 + (size_t)l * dim * hid), bxb, 0, bh3, pos_tag, l));
      } else {
        add(gemm("w1", hid, dim, wa(wo.w1 + (size_t)l * dim * hid), bxb, bh1, Rr, pos_tag, l));
        add(gemm("w3", hid, dim, wa(wo.w3 + (size_t)l * dim * hid), bxb, bh3, Rr, pos_tag, l));
      }
      add(silu(Rr, pos_tag, l));
      if (Rr == 1) add(gemv(g_2, wa(wo.w2 + (size_t)l * hid * dim), bh1, 0, bxo, pos_tag, l));
      else add(gemm("w2", dim, hid, wa(wo.w2 + (size_t)l * hid * dim), bh1, bxo, Rr, pos_tag, l));
      add(residual(Rr, pos_tag, l, "residual_ffn"));
    }
    // final rmsnorm and classifier on the last row only (run.c computes logits for the current token)
    const uint64_t last = (uint64_t)(Rr - 1) * dim * 4;
    add(rmsnorm(bx, last, wa(wo.rms_final), bxb, 0, 1, pos_tag, -1, "rmsnorm_final"));
    add(gemv(g_cls, wa(wo.wcls), bxb, 0, blog, pos_tag, -1));
    return tot;
  };

  // ---- runq.c forward on the device (q80 mode): same op order as the fp32 decode pass, plus one quantize_q80 launch
  // before each distinct matmul input (runq.c quantizes xb once for q/k/v, the attention output for wo, xb once for
  // w1/w3, hb for w2, the final-normed x for the classifier)
  auto forward_q8 = [&](int pos_tag) {
    Counters tot;
    auto add = [&](const Counters& x) { tot.cycles += x.cycles; tot.instrs += x.instrs; };
    for (int l = 0; l < L; ++l) {
      add(rmsnorm(bx, 0, W0 + qf.rms_att + (size_t)l * dim * 4, bxb, 0, 1, pos_tag, l, "rmsnorm_att"));
      add(quant(bxb, bxq, bxs, dim, pos_tag, l, "quant_x"));
      add(gemvq(g_q, qt(qf.wq, l, (size_t)dim * dim), bxq, bxs, bq, pos_tag, l));
      add(gemvq(g_k, qt(qf.wk, l, (size_t)dim * kvd), bxq, bxs, bk, pos_tag, l));
      add(gemvq(g_v, qt(qf.wv, l, (size_t)dim * kvd), bxq, bxs, bv, pos_tag, l));
      add(rope(1, l, pos_tag));
      add(attn(l, pos_tag));
      add(quant(bao, bxq, bxs, dim, pos_tag, l, "quant_ao"));
      add(gemvq(g_o, qt(qf.wo, l, (size_t)dim * dim), bxq, bxs, bxo, pos_tag, l));
      add(residual(1, pos_tag, l, "residual_att"));
      add(rmsnorm(bx, 0, W0 + qf.rms_ffn + (size_t)l * dim * 4, bxb, 0, 1, pos_tag, l, "rmsnorm_ffn"));
      add(quant(bxb, bxq, bxs, dim, pos_tag, l, "quant_x"));
      add(gemvq(g_1, qt(qf.w1, l, (size_t)dim * hid), bxq, bxs, bh1, pos_tag, l));
      add(gemvq(g_3, qt(qf.w3, l, (size_t)dim * hid), bxq, bxs, bh3, pos_tag, l));
      add(silu(1, pos_tag, l));
      add(quant(bh1, bhq, bhs, hid, pos_tag, l, "quant_h"));
      add(gemvq(g_2, qt(qf.w2, l, (size_t)hid * dim), bhq, bhs, bxo, pos_tag, l));
      add(residual(1, pos_tag, l, "residual_ffn"));
    }
    add(rmsnorm(bx, 0, W0 + qf.rms_final, bxb, 0, 1, pos_tag, -1, "rmsnorm_final"));
    add(quant(bxb, bxq, bxs, dim, pos_tag, -1, "quant_x"));
    add(gemvq(g_cls, W0 + qf.wcls, bxq, bxs, blog, pos_tag, -1));
    return tot;
  };
  // runq.c:203-205: the token embedding row is the dequantized Q8_0 row, q * s with a float32 product
  std::vector<float> emb_row(dim);   // must outlive the queued write (the queue reads it at d.finish(), not at enqueue)
  auto embed_q8 = [&](int tok) {
    std::vector<float>& row = emb_row;
    const int8_t* q = reinterpret_cast<const int8_t*>(qf.blob.data() + qf.tok) + (size_t)tok * dim;
    const float* sc = reinterpret_cast<const float*>(qf.blob.data() + qf.tok + (size_t)V * dim) + (size_t)tok * dim / GSq;
    for (int i = 0; i < dim; ++i) row[i] = (float)q[i] * sc[i / GSq];
    RT_CHECK(vx_enqueue_write(d.q, bx.h, 0, row.data(), (uint64_t)dim * 4, 0, nullptr, nullptr));
  };

  // ---- -R table: the one-time setup launch (model-load cost), then the entry-by-entry host check of the table it wrote
  if (tabm) {
    ps_rope_table_arg_t ta = {};
    ta.tab_addr = btab.addr; ta.npos = (uint32_t)S_max; ta.head_size = (uint32_t)hs; ta.theta = 10000.0f;
    const Counters sc = d.launch1d(k_rtab, "rope_table", &ta, sizeof ta, (uint32_t)(S_max * (hs / 2)), -1, -1, "rope_table");
    std::vector<float> tab(tab_n);
    d.read(tab.data(), btab, 0, tab_n * 4);
    const bool tok = ps_rope::check_table("", tab.data(), (uint32_t)hs, (uint32_t)S_max, 10000.0f);
    std::printf("PSSETUP: op=rope_table entries=%zu bytes=%zu cycles=%llu instrs=%llu amortised_over_seq_len_cycles_per_token=%.1f\n",
                tab_n / 2, tab_n * 4, (unsigned long long)sc.cycles, (unsigned long long)sc.instrs, (double)sc.cycles / S_max);
    if (!tok) die("rope_table does not match the host recomputation (see PSCHECK rope_table_vs_host / PSTABLE above)");
  }

  // ---- the position loop
  std::FILE* fo = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "wb");
  if (!out_path.empty() && !fo) die("cannot write " + out_path);
  std::vector<float> logits(V);
  std::vector<int> fed;  // tokens fed at positions 0, 1, ...
  int compared = 0, agree = 0, near_ties = 0, bad = 0, logit_fail = 0, first_tie = -1;
  bool history_golden = true, stop_compare = false;
  double max_logits_rel = 0.0;
  ps::Timer wall;
  int pos = 0;
  while (pos < npos) {
    ps::Timer tpos;
    const uint64_t launches0 = d.launches;
    Counters tot;
    int this_pos;  // the position whose logits are in blog
    if (mode == "prefill" && pos == 0) {
      std::vector<int32_t> pv(P);
      for (int i = 0; i < P; ++i) {
        pv[i] = i;
        d.copy(bx, (uint64_t)i * dim * 4, bw, (wo.emb + (size_t)seq[i] * dim) * 4, (uint64_t)dim * 4);
        fed.push_back(seq[i]);
      }
      RT_CHECK(vx_enqueue_write(d.q, bpos.h, 0, pv.data(), (uint64_t)P * 4, 0, nullptr, nullptr));
      d.finish();
      tot = forward((uint32_t)P, P - 1);
      this_pos = P - 1;
    } else {
      // tf / prefill's decode positions: the golden token; free: the token chosen at the end of the previous position
      if (mode != "free" || pos == 0) fed.push_back(seq[pos]);
      const int tok = fed[pos];
      const int32_t pv = pos;
      // run.c: x = token_embedding_table[token]
      if (q8) embed_q8(tok);
      else d.copy(bx, 0, bw, (wo.emb + (size_t)tok * dim) * 4, (uint64_t)dim * 4);
      RT_CHECK(vx_enqueue_write(d.q, bpos.h, 0, &pv, 4, 0, nullptr, nullptr));
      d.finish();
      tot = q8 ? forward_q8(pos) : forward(1, pos);
      this_pos = pos;
    }
    d.read(logits.data(), blog, 0, (uint64_t)V * 4);
    if (fo) std::fwrite(logits.data(), 4, V, fo);
    std::printf("PSDIGEST: logits_dev:p%d n=%d fnv1a64=%016llx\n", this_pos, V, (unsigned long long)ps::fnv1a64_f32(logits.data(), (size_t)V));
    int am;
    double l1, l2;
    top2(logits, am, l1, l2);
    const int p = this_pos, g = seq[p + 1];
    const double lg = logits[g], rel2 = rel_margin(l1, l2), relg = rel_margin(l1, lg);
    const bool compare = !forced[p] && !stop_compare;
    const bool ag = (am == g), tie = !ag && rel2 <= kTieRel && relg <= kTieRel;
    if (compare) {
      ++compared;
      if (ag) ++agree;
      else if (tie) { ++near_ties; if (first_tie < 0) first_tie = p; }
      else ++bad;
    }
    // logits vs the NumPy float32 reference, while the input history is the golden one
    if (!ref.empty() && history_golden && p < n_ref) {
      const float* r = ref.data() + (size_t)p * V;
      double rmax = 0.0;
      for (int i = 0; i < V; ++i) rmax = std::max(rmax, std::fabs((double)r[i]));
      std::vector<double> rr(r, r + V), sc(V, std::max(rmax, (double)FLT_MIN));
      ps::ErrStats st = ps::compare(logits.data(), rr.data(), sc.data(), V, kLogitsRel);
      ps::print_check(("logits_vs_numpy:p" + std::to_string(p)).c_str(), st);
      max_logits_rel = std::max(max_logits_rel, st.max_rel);
      if (!st.ok()) ++logit_fail;
    } else if (!ref.empty() && history_golden && p >= n_ref) {
      die("reference logits do not cover position " + std::to_string(p));
    }
    std::printf("PSTOK: pos=%d token=%d argmax=%d golden_next=%d forced=%d compared=%d top1=%.9g top2=%.9g "
                "l_golden=%.9g rel_top2=%.3e rel_golden=%.3e agree=%d near_tie=%d cycles=%llu instrs=%llu "
                "launches=%llu wall_ms=%.0f\n",
                p, fed[p], am, g, forced[p], (int)compare, l1, l2, lg, rel2, relg, (int)ag, (int)tie,
                (unsigned long long)tot.cycles, (unsigned long long)tot.instrs,
                (unsigned long long)(d.launches - launches0), tpos.ms());
    std::fflush(stdout);
    // next token
    const int next_pos = p + 1;
    if (mode == "free" && next_pos < npos) {
      const int nxt = forced[p] ? seq[p + 1] : am;
      fed.push_back(nxt);
      if (nxt != seq[p + 1]) {
        history_golden = false;  // from here on the run is no longer the golden sequence
        stop_compare = true;     // reproduce the golden text up to the first (near-tie) disagreement
      }
    }
    pos = next_pos;
  }
  if (fo) std::fclose(fo);
  if (mode == "free") {
    std::printf("PSGEN: tokens=");
    for (size_t i = 0; i < fed.size(); ++i) std::printf("%s%d", i ? "," : "", fed[i]);
    std::printf("\n");
  }
  std::printf("PSSUM: mode=%s positions=%d compared=%d agree=%d near_ties=%d first_near_tie=%d failures=%d "
              "logits_checked=%d logits_failures=%d max_logits_rel=%.3e launches=%llu flushes=%llu wall_s=%.1f\n",
              mode.c_str(), npos, compared, agree, near_ties, first_tie, bad, (int)(ref.empty() ? 0 : 1), logit_fail,
              max_logits_rel, (unsigned long long)d.launches, (unsigned long long)d.launches, wall.ms() / 1000.0);
  d.release();
  const bool ok = bad == 0 && logit_fail == 0 && compared > 0;
  if (!ok) { std::printf("FAILED!\n"); return 1; }
  std::printf("PASSED!\n");
  return 0;
}
