#pragma once
#include <linux/types.h>

#define INPUT_SIZE 8
#define HIDDEN_LAYER_1_SIZE 50
#define HIDDEN_LAYER_2_SIZE 70
#define OUTPUT_SIZE 11

typedef s64 q32_32;

extern const q32_32 W1[HIDDEN_LAYER_1_SIZE][INPUT_SIZE];
extern const q32_32 B1[HIDDEN_LAYER_1_SIZE];
extern       q32_32 A1[HIDDEN_LAYER_1_SIZE];

extern const q32_32 W2[HIDDEN_LAYER_2_SIZE][HIDDEN_LAYER_1_SIZE];
extern const q32_32 B2[HIDDEN_LAYER_2_SIZE];
extern       q32_32 A2[HIDDEN_LAYER_2_SIZE];

extern const q32_32 W3[OUTPUT_SIZE][HIDDEN_LAYER_2_SIZE];
extern const q32_32 B3[OUTPUT_SIZE];
extern       q32_32 NN_OUTPUT[OUTPUT_SIZE];

void nn_gemm_q32(const q32_32 *w, const q32_32 *x, const q32_32 *b, q32_32 *y, int rows, int cols);
void nn_tanh_q32(q32_32 *v, int n);
void nn_softmax_q32(const q32_32 *logits, q32_32 *probs, int n);
void forward(const q32_32 input[INPUT_SIZE]);

