#include "nn_policy.h"
#include <linux/kernel.h>
#include <linux/math64.h>

#define Q 16
#define ONE_Q ((q16_16)1 << Q)

#ifndef S64_MAX
#define S64_MAX  ((s64)((~(u64)0) >> 1))
#define S64_MIN  ((s64)(-S64_MAX - 1))
#endif

static inline s64 sat_s64(__int128 x) {
    if (x > (__int128)S64_MAX) return S64_MAX;
    if (x < (__int128)S64_MIN) return S64_MIN;
    return (s64)x;
}

static inline q16_16 qmul_q16_16(q16_16 a, q16_16 b) {
    s64 p = (s64)a * (s64)b;
    if (p >= 0) p += (s64)1 << (Q - 1);
    else p -= (s64)1 << (Q - 1);
    p >>= Q;
    if (p > (s64)INT_MAX) return (q16_16)INT_MAX;
    if (p < (s64)INT_MIN) return (q16_16)INT_MIN;
    return (q16_16)p;
}

void nn_gemm_q16(const q16_16 *w, const q16_16 *x, const q16_16 *b, q16_16 *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        __int128 acc = 0;
        const q16_16 *wi = w + (size_t)i * cols;
        for (int j = 0; j < cols; j++) acc += (s64)qmul_q16_16(wi[j], x[j]);
        if (b) acc += b[i];
        s64 acc64 = sat_s64(acc);
        if (acc64 > (s64)INT_MAX) acc64 = (s64)INT_MAX;
        if (acc64 < (s64)INT_MIN) acc64 = (s64)INT_MIN;
        y[i] = (q16_16)acc64;
    }
}

void nn_tanh_q16(q16_16 *v, int n) {
    for (int i = 0; i < n; i++) {
        s64 x = v[i];
        if (x > (s64)(4 * ONE_Q)) v[i] = ONE_Q;
        else if (x < (s64)(-4 * ONE_Q)) v[i] = -ONE_Q;
        else {
            s64 x2 = qmul_q16_16((q16_16)x, (q16_16)x);
            s64 num = x * ((s64)27 * ONE_Q + x2);
            s64 den = ((s64)27 * ONE_Q) + (3 * x2);
            v[i] = (q16_16)(num / den);
        }
    }
}

static inline q16_16 qexp_q16_16(q16_16 x) {
    const q16_16 MIN_X = -(8 * ONE_Q);
    if (x > 0) x = 0;
    if (x < MIN_X) x = MIN_X;
    q16_16 c1 = ONE_Q, c2 = ONE_Q, c3 = ONE_Q >> 1, c4 = ONE_Q / 6;
    q16_16 c5 = ONE_Q / 24, c6 = ONE_Q / 120;
    q16_16 y = c6;
    y = qmul_q16_16(y, x) + c5;
    y = qmul_q16_16(y, x) + c4;
    y = qmul_q16_16(y, x) + c3;
    y = qmul_q16_16(y, x) + c2;
    y = qmul_q16_16(y, x) + c1;
    return y;
}

void nn_softmax_q16(const q16_16 *logits, q16_16 *probs, int n) {
    s64 max = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max) max = logits[i];
    q16_16 exps[OUTPUT_SIZE];
    s64 sum = 0;
    for (int i = 0; i < n; i++) {
        q16_16 z = (q16_16)((s64)logits[i] - max);
        q16_16 e = qexp_q16_16(z);
        exps[i] = e;
        sum += (s64)e;
    }
    for (int i = 0; i < n; i++) {
        s64 p = ((s64)exps[i] << Q) / sum;
        if (p > (s64)INT_MAX) p = (s64)INT_MAX;
        if (p < (s64)INT_MIN) p = (s64)INT_MIN;
        probs[i] = (q16_16)p;
    }
}

void forward(const q16_16 input[INPUT_SIZE]) {
    nn_gemm_q16(&W1[0][0], input, B1, A1, HIDDEN_LAYER_1_SIZE, INPUT_SIZE);
    nn_tanh_q16(A1, HIDDEN_LAYER_1_SIZE);
    nn_gemm_q16(&W2[0][0], A1, B2, A2, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE);
    nn_tanh_q16(A2, HIDDEN_LAYER_2_SIZE);
    nn_gemm_q16(&W3[0][0], A2, B3, NN_OUTPUT, OUTPUT_SIZE, HIDDEN_LAYER_2_SIZE);
    nn_softmax_q16(NN_OUTPUT, NN_OUTPUT, OUTPUT_SIZE);
}
