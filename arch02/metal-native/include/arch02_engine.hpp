#pragma once

#include "arch02_engine.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace arch02 {

inline std::string last_error() {
    const size_t n = arch02_last_error(nullptr, 0);
    std::vector<char> buffer(n ? n : 1);
    arch02_last_error(buffer.data(), buffer.size());
    return std::string(buffer.data());
}

inline void check(int32_t status) {
    if (status != ARCH02_OK) throw std::runtime_error(last_error());
}

class Engine final {
public:
    explicit Engine(const std::string &config_json) {
        check(arch02_engine_create(config_json.c_str(), &handle_));
    }

    Engine() { check(arch02_engine_create(nullptr, &handle_)); }

    static Engine load(const std::string &checkpoint_path) {
        Arch02EngineHandle *handle = nullptr;
        check(arch02_engine_load(checkpoint_path.c_str(), &handle));
        return Engine(handle);
    }

    ~Engine() { arch02_engine_destroy(handle_); }
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    Engine(Engine &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Engine &operator=(Engine &&other) noexcept {
        if (this != &other) {
            arch02_engine_destroy(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    size_t expected_tokens() const { return arch02_engine_expected_tokens(handle_); }

    Arch02StepMetrics train(const std::vector<int32_t> &inputs,
                            const std::vector<int32_t> &targets) {
        if (inputs.size() != targets.size()) throw std::invalid_argument("token counts differ");
        Arch02StepMetrics metrics{};
        check(arch02_engine_train(handle_, inputs.data(), targets.data(), inputs.size(), &metrics));
        return metrics;
    }

    float evaluate(const std::vector<int32_t> &inputs,
                   const std::vector<int32_t> &targets) {
        if (inputs.size() != targets.size()) throw std::invalid_argument("token counts differ");
        float loss = 0.0f;
        check(arch02_engine_evaluate(handle_, inputs.data(), targets.data(), inputs.size(), &loss));
        return loss;
    }

    void save(const std::string &checkpoint_path) const {
        check(arch02_engine_save(handle_, checkpoint_path.c_str()));
    }

private:
    explicit Engine(Arch02EngineHandle *handle) : handle_(handle) {}
    Arch02EngineHandle *handle_ = nullptr;
};

} // namespace arch02
