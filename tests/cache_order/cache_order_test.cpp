// (errata 38): directed + randomized ordering test of one SimX Cache (sim/simx/mem/cache.cpp), outside the core.
//
// Build/run: scripts/cache_order_test.sh <build dir> (compiles this file with the flags of <build dir>'s last libsimx
// build, links that libsimx.so, runs it). Exit 0 = every check passed.
//
// Model. One core-side input port, so the cache admits requests in issue order (= program order). A golden copy of
// memory applies each store at ISSUE; a load must return exactly the golden sector as it was when the load was issued
// (every older store merged, no younger one). A load's footprint is its whole sector (byteen = 0, as the LSU sends
// loads), so the comparison is per sector. Memory behind the cache applies writes and samples fill data when the
// request arrives and answers fills after a random latency (out of order), so it is itself consistent. After the
// traffic drains, every sector is re-loaded through the cache (catches a lost or reordered line merge, WAW) and, for
// a write-back cache, the cache is flushed and memory compared with the golden copy. A response missing after a long
// drain window is reported as a deadlock.
//
// Directed part: the stress mixes are chosen so the two hazard shapes recur many times per seed:
//   WAR: a load misses; a younger store to the same sector is admitted while the fill rides the pipe (a
//     write-through store is not coalesced at admission) and response backpressure blocks the fill-forward drain;
//   RAW (L2, pipe deeper than 2): a store hit is deferred behind a draining chain, and a younger load to the sector
//     is already in the pipe when the deferred merge is dequeued.
// The test cannot force a single exact cycle pattern through the crossbar, so it drives 12 sectors in 2 sets of a
// 2-way cache (constant conflict misses) with bursty issue and random response backpressure, over several seeds and
// pipe depths 2, 4 and 8, and reports how often each shape occurred (from the bank's PSORDER counters when
// PS_ORDER_CHECK=1) next to the data checks.
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cache.h"
#include "mem_block_pool.h"
#include "simobject.h"

using namespace vortex;

namespace {

constexpr uint32_t BLK = VX_CFG_MEM_BLOCK_SIZE;   // = one sector, the cache data granule
constexpr uint64_t BASE = 0x20000000ull;

struct HarnessCfg {
  const char* name;
  bool wb;          // write-back (L2-like) or write-through (L1-like)
  uint16_t lat;     // bank pipeline latency
  uint8_t L, S;     // log2 line, log2 sector
  uint16_t mshr;
};

struct Pending { MemRsp rsp; uint64_t due; };

struct Harness {
  HarnessCfg cfg;
  uint32_t seed;
  Cache::Ptr cache;
  std::mt19937_64 rng;
  std::vector<uint64_t> sectors;
  std::map<uint64_t, std::array<uint8_t, BLK>> golden, memory;
  std::map<uint64_t, std::pair<uint64_t, std::array<uint8_t, BLK>>> expect;   // uuid -> (sector, data)
  std::vector<Pending> mem_pending;
  uint64_t next_uuid = 1, issued = 0, loads_checked = 0, load_mismatch = 0, final_mismatch = 0, mem_mismatch = 0;
  uint64_t n_ops = 0;
  double p_issue = 0.5, p_store = 0.5, p_rsp_stall = 0.3;
  bool draining = false, verifying = false, deadlock = false;
  size_t verify_idx = 0;
  int printed = 0;
  // directed mode: scripted (cycle, is_store, sector index) list, fixed memory latency, periodic response stall
  struct Step { uint64_t at; bool st; uint32_t sec; };
  std::vector<Step> script;
  size_t script_idx = 0;
  int fixed_lat = -1;
  uint32_t stall_period = 0;   // 0: never stall the response port; k: stall on every k-th cycle
  uint64_t t0 = 0;

  Harness(const HarnessCfg& c, uint32_t s, uint64_t ops) : cfg(c), seed(s), rng(s * 7919u + 17u), n_ops(ops) {
    Cache::Config cc{};
    cc.bypass = false;
    cc.C = uint8_t(cfg.L + 2);          // 4 lines: 2 sets x 2 ways
    cc.L = cfg.L; cc.S = cfg.S; cc.W = cfg.S; cc.A = 1; cc.B = 0;
    cc.addr_width = 32; cc.num_inputs = 1; cc.mem_ports = 1;
    cc.write_back = cfg.wb; cc.write_reponse = false; cc.mshr_size = cfg.mshr; cc.latency = cfg.lat;
    cc.repl_policy = Cache::PLRU; cc.is_llc = cfg.wb;
    std::string nm = std::string(cfg.name) + "-s" + std::to_string(seed);
    cache = Cache::Create(nm.c_str(), cc);
    // 6 lines (3 per set) x sectors per line
    const uint32_t spl = 1u << (cfg.L - cfg.S);
    for (uint32_t ln = 0; ln < 6; ++ln)
      for (uint32_t s2 = 0; s2 < spl; ++s2)
        sectors.push_back(BASE + (uint64_t(ln) << cfg.L) + (uint64_t(s2) << cfg.S));
    for (auto a : sectors) {
      std::array<uint8_t, BLK> d;
      for (uint32_t i = 0; i < BLK; ++i) d[i] = uint8_t((a >> 6) * 31 + i);
      golden[a] = d; memory[a] = d;
    }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    p_issue = 0.3 + 0.65 * u(rng);
    p_store = 0.3 + 0.4 * u(rng);
    p_rsp_stall = 0.1 + 0.6 * u(rng);
  }

  uint64_t rnd(uint64_t n) { return rng() % n; }
  bool coin(double p) { return std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p; }

  void issue_load(uint64_t a) {
    auto& in = cache->core_req_in.at(0);
    MemReq r(MemOp::LD, a, nullptr, 0, next_uuid & 0xffff, 0, next_uuid);
    in.send(r);
    expect[next_uuid] = {a, golden[a]};
    ++next_uuid; ++issued;
  }
  void issue_store(uint64_t a) {
    auto& in = cache->core_req_in.at(0);
    uint64_t be;
    if (coin(0.6)) be = 0xfull << (4 * rnd(BLK / 4));      // one aligned word (the LSU's common case)
    else { be = rng(); if (be == 0) be = 1; }               // a merged multi-lane store
    auto blk = make_mem_block();
    std::memset(blk->data(), 0, blk->size());
    auto& g = golden[a];
    for (uint32_t b = 0; b < BLK; ++b)
      if (be & (1ull << b)) { (*blk)[b] = uint8_t(rng()); g[b] = (*blk)[b]; }
    MemReq r(MemOp::ST, a, blk, be, 0, 0, next_uuid);
    in.send(r);
    ++next_uuid; ++issued;
  }

  void tick_front(uint64_t now) {
    auto& in = cache->core_req_in.at(0);
    if (!script.empty() && !draining && !verifying) {
      if (script_idx < script.size()) {
        const auto& st = script[script_idx];
        if (now >= t0 + st.at && !in.full()) {
          if (st.st) issue_store(sectors[st.sec]); else issue_load(sectors[st.sec]);
          ++script_idx;
        }
      } else {
        draining = true;
      }
    } else if (!draining && !verifying) {
      if (issued < n_ops) {
        if (!in.full() && coin(p_issue)) {
          // locality: 70 % of accesses go to one of two "hot" sectors that change slowly
          uint64_t a = sectors[(coin(0.7) ? ((now / 64 + seed) % 2) : rnd(sectors.size()))];
          if (coin(p_store)) issue_store(a); else issue_load(a);
        }
      } else {
        draining = true;
      }
    } else if (verifying && verify_idx < sectors.size() && !in.full()) {
      issue_load(sectors[verify_idx++]);
    }
    // core responses (random backpressure)
    auto& out = cache->core_rsp_out.at(0);
    const bool stall = script.empty() ? coin(p_rsp_stall) : (stall_period != 0 && (now % stall_period) == 0);
    if (!out.empty() && !stall) {
      const MemRsp rsp = out.peek();
      out.pop();
      auto it = expect.find(rsp.uuid);
      if (it == expect.end()) {
        if (printed++ < 5) std::cout << cfg.name << " s" << seed << ": unexpected response uuid " << rsp.uuid << "\n";
        ++load_mismatch;
        return;
      }
      ++loads_checked;
      bool ok = rsp.data && std::memcmp(rsp.data->data(), it->second.second.data(), BLK) == 0;
      if (!ok) {
        if (verifying) ++final_mismatch; else ++load_mismatch;
        if (printed++ < 5) {
          uint32_t b = 0;
          if (rsp.data) while (b < BLK && (*rsp.data)[b] == it->second.second[b]) ++b;
          std::cout << cfg.name << " s" << seed << ": WRONG load uuid " << rsp.uuid << " sector 0x" << std::hex
                    << it->second.first << std::dec << " first bad byte " << b << (verifying ? " (final re-read)" : "")
                    << "\n";
        }
      }
      expect.erase(it);
    }
  }

  void tick_mem(uint64_t now) {
    auto& mreq = cache->mem_req_out.at(0);
    while (!mreq.empty()) {
      MemReq r = mreq.peek();
      mreq.pop();
      uint64_t a = r.addr & ~uint64_t(BLK - 1);
      auto& m = memory[a];
      if (r.is_write()) {
        for (uint32_t b = 0; b < BLK; ++b)
          if (r.byteen & (1ull << b)) m[b] = (*r.data)[b];
      } else {
        auto blk = make_mem_block();
        std::memcpy(blk->data(), m.data(), BLK);
        mem_pending.push_back({MemRsp(r.tag, r.hart_id, r.uuid, blk), now + (fixed_lat >= 0 ? uint64_t(fixed_lat) : 3 + rnd(40))});
      }
    }
    auto& mrsp = cache->mem_rsp_in.at(0);
    for (size_t i = 0; i < mem_pending.size(); ++i) {
      if (mem_pending[i].due <= now && !mrsp.full()) {
        mrsp.send(mem_pending[i].rsp);
        mem_pending.erase(mem_pending.begin() + long(i));
        break;   // one response per cycle
      }
    }
  }

  bool quiet() const { return expect.empty() && mem_pending.empty(); }
};

void tick() { SimPlatform::instance().tick(); }

}  // namespace

int main(int argc, char** argv) {
  const uint64_t ops = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 20000;
  const uint32_t seeds = (argc > 2) ? uint32_t(std::strtoul(argv[2], nullptr, 10)) : 6;
  const uint8_t L2L = uint8_t(__builtin_ctz(VX_CFG_MEM_BLOCK_SIZE)) + 1;   // L2-like: 2 sectors per line
  const uint8_t SEC = uint8_t(__builtin_ctz(VX_CFG_MEM_BLOCK_SIZE));
  const HarnessCfg cfgs[] = {
    {"L1wt-lat2", false, 2, SEC, SEC, 8},    // the L1 dcache shape (2-entry pipe)
    {"L1wt-lat4", false, 4, SEC, SEC, 8},    // an L1 with DCACHE_SIZE >= 32 KiB (deeper pipe)
    {"L2wb-lat4", true, 4, L2L, SEC, 16},    // the PS-BASE L2 shape (write-back, 2 sectors, latency 4)
    {"L2wb-lat8", true, 8, L2L, SEC, 16},
    {"L2wb-lat4-m4", true, 4, L2L, SEC, 4},  // small MSHR: stall paths (deadlock check)
  };
  std::vector<std::unique_ptr<Harness>> hs;
  const bool directed = (argc > 3) && std::string(argv[3]) == "directed";
  if (!directed) {
    for (const auto& c : cfgs)
      for (uint32_t s = 1; s <= seeds; ++s)
        hs.emplace_back(new Harness(c, s, ops));
  } else {
    // RAW sweep (errata 38 "open RAW twin"): nL loads to sector A miss and chain; the fill returns at a fixed cycle; a
    // store S to A arrives `off` cycles around the fill, then g-1 filler loads to a warm sector W, then the younger
    // load Y to A. Stock defers S behind the draining chain; if the deferred merge enters the pipe behind Y, Y returns
    // the pre-store sector (RAW). Every (config, nL, off, g, stall) point is its own cache.
    const uint32_t A = 0, Wsec = 2;
    uint32_t id = 0;
    for (const auto& c : cfgs) {
      if (!c.wb) continue;   // the L2 (write-back) shapes: the RAW twin needs a pipe deeper than 2 and a deferred write
      for (uint32_t nL : {1u, 2u, 4u, 6u, 8u, 12u})
        for (int off = -6; off <= 24; ++off)
          for (uint32_t g = 1; g <= 4; ++g)
            for (uint32_t stall : {0u, 2u, 3u}) {
              auto* h = new Harness(c, 1000 + id++, 0);
              h->fixed_lat = 30;
              h->stall_period = stall;
              h->t0 = 300;
              h->script.push_back({0, false, Wsec});            // warm W (its fill returns long before S)
              for (uint32_t i = 0; i < nL; ++i) h->script.push_back({60 + i, false, A});
              const uint64_t ts = uint64_t(60 + 30 + 4 + off);  // around the fill's arrival at the bank
              h->script.push_back({ts, true, A});
              for (uint32_t i = 1; i < g; ++i) h->script.push_back({ts + i, false, Wsec});
              h->script.push_back({ts + g, false, A});
              hs.emplace_back(h);
            }
    }
  }

  SimPlatform::instance().reset();
  uint64_t now = 0;
  for (; now < 256; ++now) tick();   // init walk

  // traffic + drain
  const uint64_t limit = ops * 40 + 100000;
  bool all_quiet = false;
  for (; now < limit && !all_quiet; ++now) {
    for (auto& h : hs) { h->tick_front(now); h->tick_mem(now); }
    tick();
    all_quiet = true;
    for (auto& h : hs) all_quiet &= (h->draining && h->quiet());
  }
  // final re-read of every sector
  for (auto& h : hs) { h->verifying = true; h->draining = false; }
  for (uint64_t t = 0; t < 200000; ++t, ++now) {
    for (auto& h : hs) { h->tick_front(now); h->tick_mem(now); }
    tick();
    bool done = true;
    for (auto& h : hs) done &= (h->verify_idx == h->sectors.size() && h->quiet());
    if (done) break;
  }
  // write-back caches: flush and compare memory
  for (auto& h : hs) h->cache->flush_begin();   // write-through banks only print their PSORDER summary
  for (uint64_t t = 0; t < 200000; ++t, ++now) {
    for (auto& h : hs) h->tick_mem(now);
    tick();
    bool done = true;
    for (auto& h : hs) done &= (!h->cfg.wb || h->cache->flush_done());
    if (done) break;
  }
  // the last write-backs leave the bank with a registered delay: let them reach memory
  for (uint64_t t = 0; t < 64; ++t, ++now) {
    for (auto& h : hs) h->tick_mem(now);
    tick();
  }
  for (auto& h : hs) h->tick_mem(now);

  int bad = 0;
  std::map<std::string, std::array<uint64_t, 6>> agg;
  for (auto& h : hs) {
    h->deadlock = !h->quiet() || h->verify_idx != h->sectors.size();
    if (h->cfg.wb)
      for (auto a : h->sectors)
        if (h->memory[a] != h->golden[a]) ++h->mem_mismatch;
    auto& v = agg[h->cfg.name];
    v[0] += h->issued; v[1] += h->loads_checked; v[2] += h->load_mismatch; v[3] += h->final_mismatch;
    v[4] += h->mem_mismatch; v[5] += h->deadlock ? 1 : 0;
    if (h->load_mismatch || h->final_mismatch || h->mem_mismatch || h->deadlock) ++bad;
  }
  for (auto& kv : agg)
    std::cout << "CACHEORDER cfg=" << kv.first << " seeds=" << seeds << " issued=" << kv.second[0]
              << " loads_checked=" << kv.second[1] << " wrong_loads=" << kv.second[2]
              << " wrong_final_reads=" << kv.second[3] << " wrong_memory_sectors=" << kv.second[4]
              << " deadlocked_seeds=" << kv.second[5] << std::endl;
  std::cout << "CACHEORDER " << (bad ? "FAILED" : "PASSED") << " harnesses=" << hs.size() << " bad=" << bad
            << " cycles=" << now << std::endl;
  hs.clear();   // bank destructors print the PSORDER exit summaries (PS_ORDER_CHECK=1)
  return bad ? 1 : 0;
}
