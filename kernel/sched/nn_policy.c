#include "nn_policy.h"
#include <linux/kernel.h>
#include <linux/math64.h>

#define Q 32
#define ONE_Q ((q32_32)1 << Q)

#ifndef S64_MAX
#define S64_MAX  ((s64)((~(u64)0) >> 1))
#define S64_MIN  ((s64)(-S64_MAX - 1))
#endif

static inline s64 sat_s64(__int128 x) {
    if (x > (__int128)S64_MAX) return S64_MAX;
    if (x < (__int128)S64_MIN) return S64_MIN;
    return (s64)x;
}

static inline q32_32 qmul_q32_32(q32_32 a, q32_32 b) {
    __int128 p = (__int128)a * (__int128)b;
    if (p >= 0) p += (__int128)1 << (Q - 1); else p -= (__int128)1 << (Q - 1);
    p >>= Q;
    return (q32_32)sat_s64(p);
}

void nn_gemm_q32(const q32_32 *w, const q32_32 *x, const q32_32 *b, q32_32 *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        __int128 acc = 0;
        const q32_32 *wi = w + (size_t)i * cols;
        for (int j = 0; j < cols; j++) acc += (s64)qmul_q32_32(wi[j], x[j]);
        if (b) acc += b[i];
        y[i] = (q32_32)sat_s64(acc);
    }
}

void nn_tanh_q32(q32_32 *v, int n) {
    for (int i = 0; i < n; i++) {
        s64 x = v[i];
        if (x > (s64)(4 * ONE_Q)) v[i] = ONE_Q;
        else if (x < (s64)(-4 * ONE_Q)) v[i] = -ONE_Q;
        else {
            s64 x2 = qmul_q32_32((q32_32)x, (q32_32)x);
            s64 num = x * ((s64)27 * ONE_Q + x2);
            s64 den = ((s64)27 * ONE_Q) + (3 * x2);
            v[i] = (q32_32)(num / den);
        }
    }
}

static inline q32_32 qexp_q32_32(q32_32 x) {
    const q32_32 MIN_X = -(8 * ONE_Q);
    if (x > 0) x = 0;
    if (x < MIN_X) x = MIN_X;
    q32_32 c1 = ONE_Q, c2 = ONE_Q, c3 = ONE_Q >> 1, c4 = ONE_Q / 6, c5 = ONE_Q / 24, c6 = ONE_Q / 120;
    q32_32 y = c6;
    y = qmul_q32_32(y, x) + c5;
    y = qmul_q32_32(y, x) + c4;
    y = qmul_q32_32(y, x) + c3;
    y = qmul_q32_32(y, x) + c2;
    y = qmul_q32_32(y, x) + c1;
    return y;
}

void nn_softmax_q32(const q32_32 *logits, q32_32 *probs, int n) {
    s64 max = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max) max = logits[i];
    q32_32 exps[OUTPUT_SIZE];
    __int128 sum = 0;
    for (int i = 0; i < n; i++) {
        q32_32 z = (q32_32)((s64)logits[i] - max);
        q32_32 e = qexp_q32_32(z);
        exps[i] = e;
        sum += e;
    }
    s64 sum64 = sat_s64(sum);
    for (int i = 0; i < n; i++) {
        __int128 p = ((__int128)exps[i] << Q) / sum64;
        probs[i] = (q32_32)sat_s64(p);
    }
}

void forward(const q32_32 input[INPUT_SIZE]) {
    nn_gemm_q32(&W1[0][0], input, B1, A1, HIDDEN_LAYER_1_SIZE, INPUT_SIZE);
    nn_tanh_q32(A1, HIDDEN_LAYER_1_SIZE);
    nn_gemm_q32(&W2[0][0], A1, B2, A2, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE);
    nn_tanh_q32(A2, HIDDEN_LAYER_2_SIZE);
    nn_gemm_q32(&W3[0][0], A2, B3, NN_OUTPUT, OUTPUT_SIZE, HIDDEN_LAYER_2_SIZE);
    nn_softmax_q32(NN_OUTPUT, NN_OUTPUT, OUTPUT_SIZE);
}

