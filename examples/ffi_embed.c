/*
 * Minimal NORU embedding example from a non-Rust host.
 *
 * Build the library first:
 *   cargo build --release
 *
 * Then compile this example on a Unix-like system:
 *   cc -O2 -o ffi_embed examples/ffi_embed.c -L target/release -lnoru
 *
 * You may need to set LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS)
 * so the dynamic loader can find libnoru at runtime.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct NoruTrainer NoruTrainer;
typedef struct NoruWeights NoruWeights;
typedef struct NoruAccumulator NoruAccumulator;

extern const char *noru_last_error(void);
extern int noru_trainer_new(size_t feature_size, size_t accumulator_size,
                            const size_t *hidden_sizes_ptr, size_t hidden_sizes_len,
                            uint8_t activation, uint64_t seed,
                            NoruTrainer **out_handle);
extern void noru_trainer_free(NoruTrainer *handle);
extern int noru_trainer_forward(NoruTrainer *handle,
                                const uint32_t *stm_ptr, size_t stm_len,
                                const uint32_t *nstm_ptr, size_t nstm_len,
                                float *out_eval);
extern int noru_trainer_zero_grad(NoruTrainer *handle);
extern int noru_trainer_backward_bce(NoruTrainer *handle, float target);
extern int noru_trainer_adam_step(NoruTrainer *handle, float lr, float grad_scale);
extern int noru_trainer_quantize(NoruTrainer *handle, NoruWeights **out_weights);
extern int noru_accumulator_new(NoruWeights *weights, NoruAccumulator **out_acc);
extern int noru_accumulator_refresh(NoruAccumulator *acc, NoruWeights *weights,
                                    const uint32_t *stm_ptr, size_t stm_len,
                                    const uint32_t *nstm_ptr, size_t nstm_len);
extern int noru_accumulator_forward(NoruAccumulator *acc, NoruWeights *weights,
                                    int32_t *out_eval);
extern void noru_accumulator_free(NoruAccumulator *handle);
extern void noru_weights_free(NoruWeights *handle);

static void check(int rc, const char *step) {
    if (rc == 0) {
        return;
    }
    const char *msg = noru_last_error();
    fprintf(stderr, "%s failed (%d): %s\n", step, rc, msg ? msg : "<no error>");
    exit(1);
}

int main(void) {
    const size_t hidden_sizes[] = {8};
    const uint32_t stm_features[] = {1, 7, 12};
    const uint32_t nstm_features[] = {3, 9, 14};

    NoruTrainer *trainer = NULL;
    NoruWeights *weights = NULL;
    NoruAccumulator *acc = NULL;

    check(noru_trainer_new(32, 16, hidden_sizes, 1, 0, 42, &trainer),
          "noru_trainer_new");

    float fp32_eval = 0.0f;
    check(noru_trainer_forward(trainer,
                               stm_features, 3,
                               nstm_features, 3,
                               &fp32_eval),
          "noru_trainer_forward");
    check(noru_trainer_zero_grad(trainer), "noru_trainer_zero_grad");
    check(noru_trainer_backward_bce(trainer, 0.75f), "noru_trainer_backward_bce");
    check(noru_trainer_adam_step(trainer, 1e-3f, 1.0f), "noru_trainer_adam_step");

    check(noru_trainer_quantize(trainer, &weights), "noru_trainer_quantize");
    check(noru_accumulator_new(weights, &acc), "noru_accumulator_new");
    check(noru_accumulator_refresh(acc, weights,
                                   stm_features, 3,
                                   nstm_features, 3),
          "noru_accumulator_refresh");

    int32_t i16_eval = 0;
    check(noru_accumulator_forward(acc, weights, &i16_eval),
          "noru_accumulator_forward");

    printf("FP32 eval: %.3f\n", fp32_eval);
    printf("i16 eval : %d\n", i16_eval);

    noru_accumulator_free(acc);
    noru_weights_free(weights);
    noru_trainer_free(trainer);
    return 0;
}
