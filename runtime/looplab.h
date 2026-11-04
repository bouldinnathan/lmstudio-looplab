// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "ggml.h"
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace looplab {

struct config {
    int start = 0;
    int end = 0;
    int passes = 1;
    float anchor = 0.5f;
    bool layer_mode = false;
    bool cache_last = false;

    static int integer(const char * name, int fallback) {
        const char * value = std::getenv(name);
        if (!value || !*value) { return fallback; }
        size_t used = 0;
        int parsed = std::stoi(value, &used);
        if (used != std::string(value).size()) { throw std::runtime_error(name); }
        return parsed;
    }

    static config read(int layers, bool moe) {
        config c;
        c.passes = integer("LOOPLAB_PASSES", 1);
        c.start = integer("LOOPLAB_START", layers / 2 - 2);
        c.end = integer("LOOPLAB_END", c.start + 4);
        const char * beta = std::getenv("LOOPLAB_ANCHOR");
        if (beta) {
            size_t used = 0;
            c.anchor = std::stof(beta, &used);
            if (used != std::string(beta).size()) { throw std::runtime_error("LOOPLAB_ANCHOR"); }
        }
        const char * mode = std::getenv("LOOPLAB_MODE");
        const std::string m = mode ? mode : (moe ? "layer" : "block");
        if (m != "block" && m != "layer") { throw std::runtime_error("Invalid LOOPLAB_MODE"); }
        c.layer_mode = m == "layer";
        const char * cache = std::getenv("LOOPLAB_CACHE");
        const std::string policy = cache ? cache : "first";
        if (policy != "first" && policy != "last") { throw std::runtime_error("Invalid LOOPLAB_CACHE"); }
        c.cache_last = policy == "last";
        if (c.passes < 1 || c.passes > 8 || !std::isfinite(c.anchor) || c.anchor < 0 || c.anchor > 1) {
            throw std::runtime_error("Invalid LoopLab passes or anchor");
        }
        if (c.passes > 1 && (c.start < 1 || c.end <= c.start || c.end >= layers || c.end - c.start > 8)) {
            throw std::runtime_error("LoopLab requires an internal window of at most 8 layers");
        }
        if (c.passes > 1 && moe && !c.layer_mode) {
            throw std::runtime_error("LoopLab MoE support requires layer mode");
        }
        return c;
    }
};

struct state {
    bool pin_routes = false;
    std::map<int, std::pair<ggml_tensor *, ggml_tensor *>> routes;
    std::map<int, std::function<void()>> restore_kv;
};

inline thread_local state * active = nullptr;

struct scope {
    state * previous;
    explicit scope(state * current) : previous(active) { active = current; }
    ~scope() { active = previous; }
    scope(const scope &) = delete;
    scope & operator=(const scope &) = delete;
};

// The same integrator is used by native graphs and scalar conformance tests.
template<typename T, typename Forward, typename Blend>
T integrate(T input, int passes, double beta, Forward forward, Blend blend) {
    T natural = forward(input);
    T refined = blend(input, natural, 1.0 / passes);
    for (int k = 1; k < passes; ++k) {
        refined = blend(refined, forward(refined), 1.0 / passes);
    }
    return blend(refined, natural, beta);
}

template<typename Layer>
ggml_tensor * run(ggml_context * ctx, ggml_cgraph * gf, const config & c,
                 ggml_tensor * input, int layers, bool moe, Layer layer) {
    auto forward_range = [&](ggml_tensor * x, int begin, int end) {
        for (int il = begin; il < end; ++il) {
            x = layer(x, il);
            // Finish attention reads before another pass overwrites the current KV rows.
            ggml_build_forward_expand(gf, x);
        }
        return x;
    };
    if (c.passes == 1) { return forward_range(input, 0, layers); }
    auto blend = [&](ggml_tensor * x, ggml_tensor * y, double amount) {
        auto * result = ggml_add(ctx, ggml_scale(ctx, x, float(1.0 - amount)),
                                ggml_scale(ctx, y, float(amount)));
        ggml_build_forward_expand(gf, result);
        return result;
    };
    auto unit = [&](ggml_tensor * x, int begin, int end) {
        state s;
        s.pin_routes = moe;
        ggml_tensor * result;
        {
            scope guard(&s);
            result = integrate(x, c.passes, c.anchor,
                               [&](ggml_tensor * value) { return forward_range(value, begin, end); }, blend);
        }
        if (c.cache_last) {
            // The stash pass changes the cache, not the hidden state sent downstream.
            forward_range(result, begin, end);
        } else {
            // First-pass K/V are identical to a separate stash pass from the saved input.
            for (const auto & entry : s.restore_kv) { entry.second(); }
        }
        return result;
    };
    input = forward_range(input, 0, c.start);
    if (c.layer_mode) {
        for (int il = c.start; il < c.end; ++il) { input = unit(input, il, il + 1); }
    } else {
        input = unit(input, c.start, c.end);
    }
    return forward_range(input, c.end, layers);
}

} // namespace looplab
