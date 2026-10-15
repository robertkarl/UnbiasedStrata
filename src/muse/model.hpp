#pragma once
#include <memory>
#include <string>
#include <vector>

namespace strata::muse {
class Model {
public:
    Model(const std::string& path, int context, size_t cache_mib);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    // Append one token at the given position. Position zero starts a new sequence.
    std::vector<float> forward(int token, int position);
    int vocabulary() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
