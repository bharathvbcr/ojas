#ifndef ARCH02_ENGINE_H
#define ARCH02_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Arch02EngineHandle Arch02EngineHandle;

typedef struct Arch02StepMetrics {
    uint64_t completed_step;
    float loss;
    float grad_norm;
    float clip_factor;
    float lr_multiplier;
    uint64_t dispatches;
} Arch02StepMetrics;

enum {
    ARCH02_OK = 0,
    ARCH02_ERROR = -1,
    ARCH02_INVALID_ARGUMENT = -2,
};

/* NULL config_json selects the exact 128M/2k bf16 champion defaults. */
int32_t arch02_engine_create(const char *config_json, Arch02EngineHandle **out_engine);
int32_t arch02_engine_load(const char *checkpoint_path, Arch02EngineHandle **out_engine);
size_t arch02_engine_expected_tokens(const Arch02EngineHandle *engine);
int32_t arch02_engine_train(
    Arch02EngineHandle *engine,
    const int32_t *input_ids,
    const int32_t *target_ids,
    size_t token_count,
    Arch02StepMetrics *out_metrics);
int32_t arch02_engine_evaluate(
    Arch02EngineHandle *engine,
    const int32_t *input_ids,
    const int32_t *target_ids,
    size_t token_count,
    float *out_loss);
int32_t arch02_engine_save(const Arch02EngineHandle *engine, const char *checkpoint_path);
void arch02_engine_destroy(Arch02EngineHandle *engine);

/* Returns required bytes including NUL. Passing NULL/0 only queries capacity. */
size_t arch02_last_error(char *buffer, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
