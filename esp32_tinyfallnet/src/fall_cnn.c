// TinyFallNet inference in plain C.  Mirrors model.py / export_c.py:ref_forward exactly.
#include <math.h>
#include <string.h>
#include "fall_cnn.h"
#include "fall_cnn_weights.h"

// activations: largest map is 16 x 75 (conv0 out) / 24 x 38 ...  two ping-pong buffers are enough
static float bufA[16 * 75], bufB[24 * 38], feat6[6 * FALL_CNN_WIN];

// y[o][t] = relu(b[o] + sum_i sum_k W[o][i][k] * x[g*Ig+i][t*s+k-K/2]),  zero padding K/2
static int conv1d(const float *x, int C, int L, const float *W, const float *b, int O, int K, int stride, int groups, float *y) {
    const int Ig = C / groups, per = O / groups, pad = K / 2, Lo = (L + 2 * pad - K) / stride + 1;
    for (int o = 0; o < O; o++) {
        const int g = o / per;
        const float *w = W + o * Ig * K;
        for (int t = 0; t < Lo; t++) {
            float acc = b[o];
            const int p0 = t * stride - pad;
            for (int i = 0; i < Ig; i++) {
                const float *xi = x + (g * Ig + i) * L, *wi = w + i * K;
                for (int k = 0; k < K; k++) {
                    const int p = p0 + k;
                    if (p >= 0 && p < L) acc += wi[k] * xi[p];
                }
            }
            y[o * Lo + t] = acc > 0.f ? acc : 0.f;
        }
    }
    return Lo;
}

static void front(const float w[FALL_CNN_WIN][6], float *x) {
    float u[3] = {0, 0, 0};
    for (int t = 0; t < 35; t++) for (int j = 0; j < 3; j++) u[j] += w[t][j];
    for (int j = 0; j < 3; j++) u[j] /= 35.f;
    const float un = sqrtf(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) + 1e-6f;
    for (int j = 0; j < 3; j++) u[j] /= un;
    for (int t = 0; t < FALL_CNN_WIN; t++) {
        const float *a = w[t], *g = w[t] + 3;
        const float a2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2], an = sqrtf(a2);
        const float av = a[0] * u[0] + a[1] * u[1] + a[2] * u[2];
        const float g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        const float gv = g[0] * u[0] + g[1] * u[1] + g[2] * u[2];
        x[0 * FALL_CNN_WIN + t] = an - 1.f;
        x[1 * FALL_CNN_WIN + t] = av - 1.f;
        x[2 * FALL_CNN_WIN + t] = sqrtf(fmaxf(0.f, a2 - av * av));
        x[3 * FALL_CNN_WIN + t] = av / (an + 1e-6f);
        x[4 * FALL_CNN_WIN + t] = fabsf(gv) / 200.f;
        x[5 * FALL_CNN_WIN + t] = sqrtf(fmaxf(0.f, g2 - gv * gv)) / 200.f;
    }
}

#if FALL_CNN_PHYS
static float sd(const float *v, int n) {   // sample std (n-1), like torch.std
    float m = 0, q = 0;
    for (int i = 0; i < n; i++) m += v[i];
    m /= n;
    for (int i = 0; i < n; i++) q += (v[i] - m) * (v[i] - m);
    return sqrtf(q / (n - 1));
}

// 6 physics scalars, = model.py:physics()
static void physics(const float w[FALL_CNN_WIN][6], float *out) {
    static float an[FALL_CNN_WIN], gn[FALL_CNN_WIN];
    float u[3] = {0, 0, 0}, p[3] = {0, 0, 0};
    for (int t = 0; t < FALL_CNN_WIN; t++) {
        an[t] = sqrtf(w[t][0] * w[t][0] + w[t][1] * w[t][1] + w[t][2] * w[t][2]);
        gn[t] = sqrtf(w[t][3] * w[t][3] + w[t][4] * w[t][4] + w[t][5] * w[t][5]);
        for (int j = 0; j < 3; j++) { if (t < 35) u[j] += w[t][j]; if (t >= 100) p[j] += w[t][j]; }
    }
    float nu = 0, np_ = 0;
    for (int j = 0; j < 3; j++) { u[j] /= 35.f; p[j] /= 50.f; }
    nu = sqrtf(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) + 1e-6f; np_ = sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) + 1e-6f;
    out[0] = (u[0] * p[0] + u[1] * p[1] + u[2] * p[2]) / (nu * np_);
    out[1] = sd(an + 100, 50) * 5.f;
    float gm = 0; for (int t = 100; t < FALL_CNN_WIN; t++) gm += gn[t];
    out[2] = gm / 50.f / 200.f;
    float gx = 0; for (int t = 40; t < 100; t++) if (gn[t] > gx) gx = gn[t];
    out[3] = gx / 200.f;
    float amin = 1e9f; for (int t = 20; t < 50; t++) if (an[t] < amin) amin = an[t];
    out[4] = amin;
    out[5] = sd(an, 35) * 5.f;
}
#define N_FC1_IN (320 + 6)
#else
#define N_FC1_IN 320
#endif

float fall_cnn_score(const float win[FALL_CNN_WIN][6]) {
    front(win, feat6);
    int L = conv1d(feat6, 6, FALL_CNN_WIN, CONV0_W, CONV0_B, 16, 5, 2, 1, bufA);   // 16 x 75
    L = conv1d(bufA, 16, L, DW1_W, DW1_B, 16, 5, 2, 16, bufB);                       // 16 x 38
    L = conv1d(bufB, 16, L, PW1_W, PW1_B, 24, 1, 1, 1, bufA);                        // 24 x 38
    L = conv1d(bufA, 24, L, DW2_W, DW2_B, 24, 5, 2, 24, bufB);                       // 24 x 19
    L = conv1d(bufB, 24, L, PW2_W, PW2_B, 32, 1, 1, 1, bufA);                        // 32 x 19
    L = conv1d(bufA, 32, L, DW3_W, DW3_B, 32, 5, 2, 32, bufB);                       // 32 x 10
    L = conv1d(bufB, 32, L, PW3_W, PW3_B, 32, 1, 1, 1, bufA);                        // 32 x 10 -> flatten 320 (channel-major, = torch Flatten)
#if FALL_CNN_PHYS
    physics(win, bufA + 320);                                                         // flatten(320) ++ 6 physics scalars
#endif
    float h[16];
    for (int o = 0; o < 16; o++) {
        float acc = FC1_B[o];
        for (int i = 0; i < N_FC1_IN; i++) acc += FC1_W[o * N_FC1_IN + i] * bufA[i];
        h[o] = acc > 0.f ? acc : 0.f;
    }
    float z = FC2_B[0];
    for (int i = 0; i < 16; i++) z += FC2_W[i] * h[i];
    return 1.f / (1.f + expf(-z));
}

float fall_cnn_threshold(void) { return FALL_CNN_THRESHOLD; }

float fall_cnn_selftest(void) {
    return fabsf(fall_cnn_score((const float (*)[6])SELFTEST_WINDOW) - SELFTEST_EXPECTED);
}
