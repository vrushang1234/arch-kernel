#pragma once
#include <linux/types.h>

#define INPUT_SIZE 8
#define HIDDEN_LAYER_1_SIZE 50
#define HIDDEN_LAYER_2_SIZE 70
#define OUTPUT_SIZE 11

typedef s32 q16_16;

extern const q16_16 W1[HIDDEN_LAYER_1_SIZE][INPUT_SIZE];
extern const q16_16 B1[HIDDEN_LAYER_1_SIZE];
extern       q16_16 A1[HIDDEN_LAYER_1_SIZE];

extern const q16_16 W2[HIDDEN_LAYER_2_SIZE][HIDDEN_LAYER_1_SIZE];
extern const q16_16 B2[HIDDEN_LAYER_2_SIZE];
extern       q16_16 A2[HIDDEN_LAYER_2_SIZE];

extern const q16_16 W3[OUTPUT_SIZE][HIDDEN_LAYER_2_SIZE];
extern const q16_16 B3[OUTPUT_SIZE];
extern       q16_16 NN_OUTPUT[OUTPUT_SIZE];

void nn_gemm_q16(const q16_16 *w, const q16_16 *x, const q16_16 *b, q16_16 *y, int rows, int cols);
void nn_tanh_q16(q16_16 *v, int n);
void nn_softmax_q16(const q16_16 *logits, q16_16 *probs, int n);
void forward(const q16_16 input[INPUT_SIZE]);

