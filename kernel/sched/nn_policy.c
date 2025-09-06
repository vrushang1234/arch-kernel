#include "nn_policy.h"
#include <linux/kernel.h>
#include <linux/math64.h>
#include <linux/time.h>

#define Q 32
#define ONE_Q ((q32_32)1 << Q)

#ifndef S64_MAX
#define S64_MAX  ((s64)((~(u64)0) >> 1))
#define S64_MIN  ((s64)(-S64_MAX - 1))
#endif

q32_32 A1[HIDDEN_LAYER_1_SIZE];
q32_32 A2[HIDDEN_LAYER_2_SIZE];
q32_32 NN_OUTPUT[OUTPUT_SIZE];

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
    u64 sum = 0;
    for (int i = 0; i < n; i++) {
        q32_32 z = (q32_32)((s64)logits[i] - max);
        q32_32 e = qexp_q32_32(z);
        exps[i] = e;
        sum += (u64)e;
    }
    if (!sum) { /* extremely defensive; should not happen */
        for (int i = 0; i < n; i++) probs[i] = 0;
        return;
    }
    for (int i = 0; i < n; i++) {
        u64 q = mul_u64_u64_div_u64((u64)exps[i], (u64)ONE_Q, (u64)sum);
        probs[i] = (q32_32)(s64)q;
    }
}

void forward(const q32_32 input[INPUT_SIZE]) {
    nn_gemm_q32(&W1[0][0], input, B1, A1, HIDDEN_LAYER_1_SIZE, INPUT_SIZE);
    nn_tanh_q32(A1, HIDDEN_LAYER_1_SIZE);
    nn_gemm_q32(&W2[0][0], A1, B2, A2, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE);
    nn_tanh_q32(A2, HIDDEN_LAYER_2_SIZE);
    nn_gemm_q32(&W3[0][0], A2, B3, NN_OUTPUT, OUTPUT_SIZE, HIDDEN_LAYER_2_SIZE);
}

static inline q32_32 q32_from_ratio_s64(s64 num, s64 den)
{
    if (unlikely(den == 0))
        return (num >= 0) ? (q32_32)S64_MAX : (q32_32)S64_MIN;

    int neg = ((num < 0) ^ (den < 0));
    u64 a = (num < 0) ? (u64)(-(num + 1)) + 1 : (u64)num;
    u64 b = (den < 0) ? (u64)(-(den + 1)) + 1 : (u64)den;

    u64 q = mul_u64_u64_div_u64(a, (u64)ONE_Q, b);  /* floor((a*2^32)/b) */

    if (neg) {
        if (q > (u64)S64_MAX + 1) return (q32_32)S64_MIN;
        return (q32_32)-(s64)q;
    } else {
        if (q > (u64)S64_MAX) return (q32_32)S64_MAX;
        return (q32_32)(s64)q;
    }
}

static inline q32_32 q32_from_ns(s64 ns) { return q32_from_ratio_s64(ns, (s64)NSEC_PER_SEC); }
static inline q32_32 q32_from_ns_u64(u64 ns) { return q32_from_ns((s64)ns); }
static inline q32_32 q32_from_int(s64 x) { __int128 t = (__int128)x << Q; return (q32_32)sat_s64(t); }

unsigned int rl_decide(u64 task_last_wait_time,
              u64 task_total_wait_time,   u64 task_wait_count,
              u64 last_burst_time,        u64 total_burst_time, u64 task_burst_count,
              u64 task_vruntime,          u64 task_sum_exec_runtime,
              u64 queue_total_wait_time,  u64 queue_wait_count,
              u64 queue_total_burst_time, u64 total_burst_count)
{
    q32_32 in[INPUT_SIZE];
    in[0] = q32_from_ns_u64(task_last_wait_time);
    in[1] = task_wait_count
          ? q32_from_ratio_s64((s64)task_total_wait_time, (s64)task_wait_count * (s64)NSEC_PER_SEC)
          : 0;
    in[2] = q32_from_ns_u64(last_burst_time);
    in[3] = task_burst_count
          ? q32_from_ratio_s64((s64)total_burst_time, (s64)task_burst_count * (s64)NSEC_PER_SEC)
          : 0;
    in[4] = q32_from_ns_u64(task_vruntime);
    in[5] = q32_from_ns_u64(task_sum_exec_runtime);
    in[6] = queue_wait_count
          ? q32_from_ratio_s64((s64)queue_total_wait_time, (s64)queue_wait_count * (s64)NSEC_PER_SEC)
          : 0;
    in[7] = total_burst_count
          ? q32_from_ratio_s64((s64)queue_total_burst_time, (s64)total_burst_count * (s64)NSEC_PER_SEC)
          : 0;
    forward(in);
    int argmax = 0;
    q32_32 best = NN_OUTPUT[0];
    for (int i = 1; i < OUTPUT_SIZE; ++i)
        if (NN_OUTPUT[i] > best) { best = NN_OUTPUT[i]; argmax = i; }
    return slice_values[argmax];
}

