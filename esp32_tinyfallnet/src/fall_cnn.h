// TinyFallNet - 1D-CNN fall classifier for one impact candidate (ESP32 / any MCU, float32, no library).
//
// Usage: keep a ring buffer of the last 150 samples (50 Hz).  When |a| has a local max >= 1.6 g at sample k,
// wait until k+100 has arrived, copy samples [k-50, k+100) into win[150][6] = {ax,ay,az (g), gx,gy,gz (deg/s)}
// and call fall_cnn_score(win).  Fall if score >= FALL_CNN_THRESHOLD.
// Accelerometer values must be clipped to +-2 g (the model was trained that way; a no-op on a +-2 g sensor).
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#define FALL_CNN_WIN 150

float fall_cnn_score(const float win[FALL_CNN_WIN][6]);   // probability 0..1
float fall_cnn_threshold(void);
float fall_cnn_selftest(void);                             // |score - expected| on the built-in test window (should be < 1e-4)

#ifdef __cplusplus
}
#endif
