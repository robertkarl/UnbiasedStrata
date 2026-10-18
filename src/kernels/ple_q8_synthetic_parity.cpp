// UnbiasedStrata: Q8_0 PLE rows of a synthetic table against ggml's reference dequantizer, through every reader
// (direct, mapped, and mapped with the table in RAM). The program writes the table itself, so it needs no model file.
//
//     ple_q8_synthetic_parity <dir>
//
// The table is a PLE-only GGUF with `kRows` Q8_0 rows of 170 bytes, quantized by ggml from random values. Most rows
// cross a 4 KiB page boundary at some offset, so the direct reader's page arithmetic is tested with 170-byte rows.
#define NOMINMAX
#include "strata/kernels/ngram.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

struct ReaderCase {
    const char* name;
    k::PleIo mode;
    bool lock;   // `--ple-io ram`
};

const ReaderCase kReaderCases[] = {
    {"direct", k::PleIo::Direct, false},
    {"mmap", k::PleIo::Mmap, false},
    {"ram", k::PleIo::Mmap, true},
};

constexpr uint32_t kRows = 8192;

std::string le(uint64_t value, int bytes) {
    std::string s((size_t) bytes, '\0');
    for (int i = 0; i < bytes; ++i) s[(size_t) i] = (char) ((value >> (8 * i)) & 0xff);
    return s;
}

std::string gguf_string(const char* s) { return le(std::strlen(s), 8) + s; }

// Random values quantized by ggml's reference Q8_0 quantizer. Row 0 is all zeros (scale 0).
std::vector<uint8_t> make_rows(const ggml_type_traits* traits) {
    std::mt19937 rng(8);
    std::normal_distribution<float> value(0.0f, 0.05f);
    std::vector<uint8_t> bytes((size_t) kRows * k::PLE_ROW_BYTES_Q8_0);
    float row[k::PLE_HEAD_DIM];
    for (uint32_t r = 0; r < kRows; ++r) {
        for (float& v : row) v = r == 0 ? 0.0f : value(rng) * (float) (1 + r % 7);
        traits->from_float_ref(row, bytes.data() + (size_t) r * k::PLE_ROW_BYTES_Q8_0, k::PLE_HEAD_DIM);
    }
    return bytes;
}

bool write_table(const std::string& path, const std::vector<uint8_t>& rows) {
    std::string head = "GGUF" + le(3, 4) + le(1, 8) + le(1, 8);
    head += gguf_string("general.architecture") + le(8, 4) + gguf_string("strata-ple");
    head += gguf_string("per_layer_token_embd.weight") + le(2, 4) + le(k::PLE_HEAD_DIM, 8) + le(kRows, 8);
    head += le(GGML_TYPE_Q8_0, 4) + le(0, 8);
    while (head.size() % 32) head += '\0';
    std::ofstream out(path, std::ios::binary);
    out.write(head.data(), (std::streamsize) head.size());
    out.write((const char*) rows.data(), (std::streamsize) rows.size());
    return (bool) out;
}

// Returns the count of rows that differ from ggml's dequantization of the same bytes: every row through
// `gather_batch` (the prompt path), and through `read_row` and `gather` (the token path).
int check(const std::string& path, const ReaderCase& c, const std::vector<uint8_t>& bytes, const ggml_type_traits* traits) {
    k::PleIoOptions io;
    io.mode = c.mode;
    io.lock = c.lock;
    io.cache_rows = 0;
    k::PleTable table;
    std::string err;
    if (!table.open(path, err, io)) {
        std::fprintf(stderr, "%s: open: %s\n", c.name, err.c_str());
        return 1;
    }
    if (std::strcmp(table.format(), "Q8_0") != 0 || table.rows() != kRows) {
        std::fprintf(stderr, "%s: format %s, %llu rows\n", c.name, table.format(), (unsigned long long) table.rows());
        return 1;
    }
    std::vector<float> want((size_t) kRows * k::PLE_HEAD_DIM), got(want.size());
    traits->to_float(bytes.data(), want.data(), (int64_t) want.size());
    std::vector<uint32_t> ids(kRows);
    for (uint32_t r = 0; r < kRows; ++r) ids[r] = r;
    if (!table.gather_batch(ids.data(), kRows / k::PLE_N_HEADS, got.data(), err)) {
        std::fprintf(stderr, "%s: gather_batch: %s\n", c.name, err.c_str());
        return 1;
    }
    int bad = 0;
    const size_t row_floats = (size_t) k::PLE_HEAD_DIM * sizeof(float);
    for (uint32_t r = 0; r < kRows; ++r)
        bad += std::memcmp(&got[(size_t) r * k::PLE_HEAD_DIM], &want[(size_t) r * k::PLE_HEAD_DIM], row_floats) != 0;
    const uint32_t probes[k::PLE_N_HEADS] = {0, 1, 23, 24, 25, 47, 48, 1000, 2047, 2048, 4095, 4096, 5000, 7000, 8190, kRows - 1};
    float row[k::PLE_HEAD_DIM], token[k::PLE_N_HEADS * k::PLE_HEAD_DIM];
    table.gather(probes, token);
    for (int h = 0; h < k::PLE_N_HEADS; ++h) {
        table.read_row(probes[h], row);
        bad += std::memcmp(row, &want[(size_t) probes[h] * k::PLE_HEAD_DIM], row_floats) != 0;
        bad += std::memcmp(&token[(size_t) h * k::PLE_HEAD_DIM], &want[(size_t) probes[h] * k::PLE_HEAD_DIM], row_floats) != 0;
    }
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ple_q8_synthetic_parity <dir>\n");
        return 2;
    }
    const ggml_type_traits* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    const std::vector<uint8_t> bytes = make_rows(traits);
    const std::string path = std::string(argv[1]) + "/ple_q8_synthetic_parity.gguf";
    if (!write_table(path, bytes)) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return 1;
    }
    int failed = 0;
    for (const ReaderCase& c : kReaderCases) {
        const int bad = check(path, c, bytes, traits);
        std::printf("Q8_0 PLE, %s reader: %u rows, %d differ from ggml %s\n", c.name, kRows, bad, bad == 0 ? "PASS" : "FAIL");
        failed += bad != 0;
    }
    std::filesystem::remove(path);
    return failed == 0 ? 0 : 1;
}
