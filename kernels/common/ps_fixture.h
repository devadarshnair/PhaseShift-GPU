// ps_fixture.h -- host-side reader for PhaseShift `ps-fixture-v1` fixtures.
//
// Format (written by model/make_fixtures.py): a fixture directory holds `meta.json`
//   {"format": "ps-fixture-v1", "model", "op", "layer", "pos", "params": {...},
//    "tensors": {"<name>": {"file": "<path relative to the dir>", "shape": [...], "dtype": "<f4",
//                           "role": "in"|"out"|"weight", "sha256": "<hex>"}}}
// and one raw little-endian float32 file per tensor (C order, no header).
//
// This header is self-contained (C++17, no third-party JSON library: the plan installs nothing): a small
// recursive-descent JSON parser, a SHA-256 (FIPS 180-4) to verify each tensor file against meta.json, and
// `ps::Fixture` with `tensor(name)` returning the data + shape. Every check failure throws ps::FixtureError, so a
// corrupt or mismatched fixture can never be read silently. Host-only (never included by a device kernel).
#ifndef PS_FIXTURE_H
#define PS_FIXTURE_H

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps {

struct FixtureError : std::runtime_error {
  explicit FixtureError(const std::string& m) : std::runtime_error(m) {}
};

// ------------------------------------------------------------------------------------------------ JSON value
struct Json {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  bool has(const std::string& k) const { return kind == Object && obj.count(k) != 0; }
  const Json& at(const std::string& k) const {
    if (kind != Object) throw FixtureError("json: not an object (looking up '" + k + "')");
    auto it = obj.find(k);
    if (it == obj.end()) throw FixtureError("json: missing key '" + k + "'");
    return it->second;
  }
  const std::string& as_str() const {
    if (kind != String) throw FixtureError("json: not a string");
    return str;
  }
  double as_num() const {
    if (kind != Number) throw FixtureError("json: not a number");
    return num;
  }
};

class JsonParser {
public:
  explicit JsonParser(const std::string& s) : s_(s) {}
  Json parse() {
    Json v = value();
    ws();
    if (i_ != s_.size()) fail("trailing characters");
    return v;
  }

private:
  const std::string& s_;
  size_t i_ = 0;

  [[noreturn]] void fail(const char* what) const {
    throw FixtureError(std::string("json: ") + what + " at offset " + std::to_string(i_));
  }
  void ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_; }
  char peek() { ws(); if (i_ >= s_.size()) fail("unexpected end"); return s_[i_]; }
  void expect(char c) { if (peek() != c) fail("unexpected character"); ++i_; }
  bool lit(const char* w) {
    size_t n = std::strlen(w);
    if (s_.compare(i_, n, w) == 0) { i_ += n; return true; }
    return false;
  }
  Json value() {
    char c = peek();
    Json v;
    if (c == '{') {
      v.kind = Json::Object; ++i_;
      if (peek() == '}') { ++i_; return v; }
      while (true) {
        if (peek() != '"') fail("expected key");
        std::string k = string();
        expect(':');
        if (v.obj.count(k)) fail("duplicate key");
        v.obj[k] = value();
        if (peek() == ',') { ++i_; continue; }
        expect('}');
        return v;
      }
    }
    if (c == '[') {
      v.kind = Json::Array; ++i_;
      if (peek() == ']') { ++i_; return v; }
      while (true) {
        v.arr.push_back(value());
        if (peek() == ',') { ++i_; continue; }
        expect(']');
        return v;
      }
    }
    if (c == '"') { v.kind = Json::String; v.str = string(); return v; }
    if (lit("true")) { v.kind = Json::Bool; v.b = true; return v; }
    if (lit("false")) { v.kind = Json::Bool; v.b = false; return v; }
    if (lit("null")) { return v; }
    // number, strict JSON grammar: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?
    size_t st = i_;
    auto digit = [&]() { return i_ < s_.size() && std::isdigit((unsigned char)s_[i_]); };
    auto digits = [&]() { size_t d = i_; while (digit()) ++i_; return i_ > d; };
    if (i_ < s_.size() && s_[i_] == '-') ++i_;
    if (i_ < s_.size() && s_[i_] == '0') ++i_;
    else if (!digits()) fail("bad value");
    if (i_ < s_.size() && s_[i_] == '.') { ++i_; if (!digits()) fail("bad number"); }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      if (!digits()) fail("bad number");
    }
    v.kind = Json::Number;
    try { v.num = std::stod(s_.substr(st, i_ - st)); } catch (...) { fail("bad number"); }
    return v;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (i_ >= s_.size()) fail("unterminated string");
      char c = s_[i_++];
      if (c == '"') return out;
      if (c != '\\') { out.push_back(c); continue; }
      if (i_ >= s_.size()) fail("bad escape");
      char e = s_[i_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {  // keep ASCII code points; fixtures only contain ASCII keys/paths
          if (i_ + 4 > s_.size()) fail("bad \\u escape");
          unsigned cp = (unsigned)std::stoul(s_.substr(i_, 4), nullptr, 16);
          i_ += 4;
          if (cp > 0x7f) fail("non-ASCII \\u escape not supported");
          out.push_back((char)cp);
          break;
        }
        default: fail("bad escape");
      }
    }
  }
};

// ------------------------------------------------------------------------------------------------ SHA-256
// FIPS 180-4 SHA-256, used only to verify fixture files against meta.json (same check as load_fixture in Python).
class Sha256 {
public:
  Sha256() { reset(); }
  void update(const uint8_t* p, size_t n) {
    len_ += n;
    while (n > 0) {
      size_t k = std::min(n, (size_t)64 - fill_);
      std::memcpy(buf_ + fill_, p, k);
      fill_ += k; p += k; n -= k;
      if (fill_ == 64) { block(buf_); fill_ = 0; }
    }
  }
  std::string hex() {
    uint64_t bits = len_ * 8;
    uint8_t pad = 0x80, zero = 0;
    update(&pad, 1);
    while (fill_ != 56) update(&zero, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; ++i) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(lb, 8);
    static const char* hx = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 8; ++i)
      for (int j = 3; j >= 0; --j) {
        uint8_t byte = (uint8_t)(h_[i] >> (8 * j));
        out.push_back(hx[byte >> 4]); out.push_back(hx[byte & 15]);
      }
    reset();
    return out;
  }

private:
  uint32_t h_[8];
  uint8_t buf_[64];
  size_t fill_;
  uint64_t len_;
  void reset() {
    static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h_, init, sizeof(h_));
    fill_ = 0; len_ = 0;
  }
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) | ((uint32_t)p[4 * i + 2] << 8) | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + mj;
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
  }
};

inline std::string sha256_hex(const std::vector<uint8_t>& data) {
  Sha256 s;
  s.update(data.data(), data.size());
  return s.hex();
}

inline std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw FixtureError("cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ------------------------------------------------------------------------------------------------ fixture
struct Tensor {
  std::string name, file, role;
  std::vector<uint32_t> shape;
  std::vector<float> data;
  size_t numel() const { size_t n = 1; for (auto d : shape) n *= d; return n; }
  uint32_t dim(size_t i) const {
    if (i >= shape.size()) throw FixtureError("tensor " + name + ": no dim " + std::to_string(i));
    return shape[i];
  }
};

class Fixture {
public:
  explicit Fixture(const std::string& dir) : dir_(dir) {
    std::vector<uint8_t> raw = read_file(dir_ + "/meta.json");
    meta_ = JsonParser(std::string(raw.begin(), raw.end())).parse();
    if (meta_.at("format").as_str() != "ps-fixture-v1")
      throw FixtureError(dir_ + ": format is not ps-fixture-v1");
  }
  const Json& meta() const { return meta_; }
  const std::string& dir() const { return dir_; }
  std::string op() const { return meta_.at("op").as_str(); }
  std::string model() const { return meta_.at("model").as_str(); }

  // Loads tensor `name`: checks dtype "<f4", file size == prod(shape) * 4 and sha256 == meta.json's.
  Tensor tensor(const std::string& name) const {
    const Json& t = meta_.at("tensors").at(name);
    Tensor out;
    out.name = name;
    out.file = t.at("file").as_str();
    out.role = t.has("role") ? t.at("role").as_str() : "";
    if (t.at("dtype").as_str() != "<f4") throw FixtureError(name + ": dtype is not <f4");
    const Json& sh = t.at("shape");
    if (sh.kind != Json::Array) throw FixtureError(name + ": shape is not an array");
    for (auto& d : sh.arr) {  // integer in [0, 2^31): no negative, fractional or huge dims
      double x = d.as_num();
      if (!(x >= 0.0 && x < 2147483648.0) || x != (double)(uint32_t)x)
        throw FixtureError(name + ": shape entry is not an integer in [0, 2^31)");
      out.shape.push_back((uint32_t)x);
    }
    size_t total = 1;
    for (auto d : out.shape) {
      if (d != 0 && total > ((size_t)1 << 40) / d) throw FixtureError(name + ": tensor too large");
      total *= d;
    }
    std::vector<uint8_t> raw = read_file(dir_ + "/" + out.file);
    if (raw.size() != out.numel() * sizeof(float))
      throw FixtureError(name + ": file size " + std::to_string(raw.size()) + " != prod(shape)*4");
    if (sha256_hex(raw) != t.at("sha256").as_str()) throw FixtureError(name + ": sha256 mismatch (" + out.file + ")");
    out.data.resize(out.numel());
    std::memcpy(out.data.data(), raw.data(), raw.size());  // host is little-endian x86_64, as the format
    return out;
  }

private:
  std::string dir_;
  Json meta_;
};

}  // namespace ps

#endif  // PS_FIXTURE_H
