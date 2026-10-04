/*
 * ==============================================================================
 * ESP32 Edge Fall Detection Node - Hybrid Edge-to-Host Architecture
 * ==============================================================================
 * Hardware:
 *   - ESP32 Development Board (ESP32-WROOM-32 / NodeMCU-32S)
 *   - IMU Sensor: MPU6050 / MPU9250 (I2C)
 *   - Optional: Active Buzzer / LED on Pin 2
 *
 * Wiring:
 *   ESP32 Pin 21 (SDA)  <---> MPU6050 SDA
 *   ESP32 Pin 22 (SCL)  <---> MPU6050 SCL
 *   ESP32 3.3V          <---> MPU6050 VCC
 *   ESP32 GND           <---> MPU6050 GND
 *   ESP32 Pin 2 (LED)   <---> On-board LED / Status Indicator
 *
 * Sampling:
 *   50 Hz (exact dt = 20 ms)
 *
 * Operation Modes:
 *   Mode 1: STREAM (Default) - Continuously sends CSV tuples:
 *           timestamp_ms,accX_g,accY_g,accZ_g,gyrX_dps,gyrY_dps,gyrZ_dps
 *   Mode 2: EVENT_BURST (Ultra Low-Power) - Runs edge impact trigger (|a| >= 1.6g),
 *           records 100 samples (+2.0s), computes quick physics tilt check,
 *           and only sends the 150-sample event window to Host if suspected.
 *
 *   Commands over Serial (115200 baud):
 *     's' -> Switch to STREAM mode
 *     'b' -> Switch to EVENT_BURST mode
 *     't' -> Trigger test alert
 * ==============================================================================
 */

#include <Wire.h>
#include <math.h>

// --- Configuration ---
#define SERIAL_BAUD       115200
#define SAMPLE_INTERVAL_MS 20      // 50 Hz = 20 ms
#define MPU_ADDR          0x68
#define LED_PIN           2

// --- Buffer Config (3.0s window: 50 pre-impact, 100 post-impact) ---
#define PRE_SAMPLES       50
#define POST_SAMPLES      100
#define TOTAL_SAMPLES     150

#define IMPACT_THRESHOLD_G 1.6f    // Trigger threshold |a| >= 1.6g
#define EDGE_TILT_GATE_DEG 25.0f   // Permissive edge gate for burst mode

// MPU6050 Sensitivity Factors (+-8g, +-2000 dps)
#define ACCEL_SCALE_8G    4096.0f  // LSB / g
#define GYRO_SCALE_2000   16.4f    // LSB / (deg/s)

// --- Sample Structure ---
struct ImuSample {
    uint32_t t_ms;
    float ax, ay, az; // in g
    float gx, gy, gz; // in dps
};

// Circular Ring Buffer
ImuSample ringBuffer[TOTAL_SAMPLES];
int bufHead = 0;
int sampleCount = 0;

// Operating Modes
enum NodeMode { MODE_STREAM = 0, MODE_BURST = 1 };
NodeMode currentMode = MODE_STREAM;

// State Machine for Burst Mode
enum DetectState { STATE_MONITORING, STATE_RECORDING_POST };
DetectState burstState = STATE_MONITORING;
int postSamplesRemaining = 0;
int impactIndex = 0;

unsigned long lastSampleTime = 0;

// --- Function Prototypes ---
void initMPU6050();
bool readMPU6050(ImuSample &sample);
float computeSVM(float ax, float ay, float az);
float computeTiltDeg(int k_impact);
void transmitBurstWindow();

void setup() {
    Serial.begin(SERIAL_BAUD);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Wire.begin(21, 22); // SDA = 21, SCL = 22 for ESP32
    Wire.setClock(400000); // 400 kHz fast I2C

    delay(200);
    initMPU6050();

    Serial.println("# [ESP32] Node Initialized at 50Hz (20ms/sample)");
    Serial.println("# [ESP32] Mode: STREAM (Send 'b' for Low-Power Burst, 's' for Stream)");
    Serial.println("timestamp_ms,accX_g,accY_g,accZ_g,gyrX_dps,gyrY_dps,gyrZ_dps");

    lastSampleTime = millis();
}

void loop() {
    // 1. Check for incoming Serial commands from Host
    if (Serial.available()) {
        char cmd = Serial.read();
        if (cmd == 's' || cmd == 'S') {
            currentMode = MODE_STREAM;
            burstState = STATE_MONITORING;
            digitalWrite(LED_PIN, LOW);
            Serial.println("# [ESP32] Switched to MODE_STREAM");
        } else if (cmd == 'b' || cmd == 'B') {
            currentMode = MODE_BURST;
            burstState = STATE_MONITORING;
            digitalWrite(LED_PIN, LOW);
            Serial.println("# [ESP32] Switched to MODE_BURST (Low-Power Trigger Mode)");
        } else if (cmd == 't' || cmd == 'T') {
            Serial.println("# [ESP32] Manual Test Trigger Sent");
        }
    }

    // 2. Exact 50 Hz Timing Loop
    unsigned long now = millis();
    if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime += SAMPLE_INTERVAL_MS;
        if (now - lastSampleTime > 100) {
            // Clock drift recovery
            lastSampleTime = now;
        }

        ImuSample s;
        s.t_ms = now;
        if (!readMPU6050(s)) return;

        // Push into Circular Buffer
        ringBuffer[bufHead] = s;
        int currentIdx = bufHead;
        bufHead = (bufHead + 1) % TOTAL_SAMPLES;
        if (sampleCount < TOTAL_SAMPLES) sampleCount++;

        float svm = computeSVM(s.ax, s.ay, s.az);

        // Mode 1: Continuous Streaming to Host
        if (currentMode == MODE_STREAM) {
            Serial.print(s.t_ms); Serial.print(",");
            Serial.print(s.ax, 4); Serial.print(",");
            Serial.print(s.ay, 4); Serial.print(",");
            Serial.print(s.az, 4); Serial.print(",");
            Serial.print(s.gx, 2); Serial.print(",");
            Serial.print(s.gy, 2); Serial.print(",");
            Serial.println(s.gz, 2);

            if (svm >= IMPACT_THRESHOLD_G) {
                digitalWrite(LED_PIN, HIGH);
            } else {
                digitalWrite(LED_PIN, LOW);
            }
        }
        // Mode 2: Hybrid Low-Power Edge Burst Mode
        else if (currentMode == MODE_BURST) {
            if (burstState == STATE_MONITORING) {
                if (sampleCount >= TOTAL_SAMPLES && svm >= IMPACT_THRESHOLD_G) {
                    // Impact event triggered! Start capturing post-impact window
                    burstState = STATE_RECORDING_POST;
                    postSamplesRemaining = POST_SAMPLES;
                    impactIndex = currentIdx;
                    digitalWrite(LED_PIN, HIGH);
                }
            } else if (burstState == STATE_RECORDING_POST) {
                postSamplesRemaining--;
                if (postSamplesRemaining <= 0) {
                    // Window fully captured (-50 to +100 samples)
                    float tilt = computeTiltDeg(impactIndex);
                    if (tilt >= EDGE_TILT_GATE_DEG) {
                        // Pass edge physics gate -> transmit 150-sample burst to Host
                        transmitBurstWindow();
                    } else {
                        Serial.print("# [EDGE_REJECT] Impact |a|=");
                        Serial.print(computeSVM(ringBuffer[impactIndex].ax, ringBuffer[impactIndex].ay, ringBuffer[impactIndex].az), 2);
                        Serial.print("g, Tilt=");
                        Serial.print(tilt, 1);
                        Serial.println(" deg (Rejected at Edge)");
                    }
                    burstState = STATE_MONITORING;
                    digitalWrite(LED_PIN, LOW);
                }
            }
        }
    }
}

// --- Direct MPU6050 Register Initialization (Zero external dependencies) ---
void initMPU6050() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B); // PWR_MGMT_1
    Wire.write(0x00); // Wake up
    Wire.endTransmission(true);
    delay(50);

    // Accel Config: +-8g (0x10)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1C);
    Wire.write(0x10);
    Wire.endTransmission(true);

    // Gyro Config: +-2000 dps (0x18)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1B);
    Wire.write(0x18);
    Wire.endTransmission(true);

    // Low-pass filter ~42Hz
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A);
    Wire.write(0x03);
    Wire.endTransmission(true);
}

// --- Read 14 bytes (Accel + Temp + Gyro) ---
bool readMPU6050(ImuSample &s) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B); // Starting register for Accel X
    if (Wire.endTransmission(false) != 0) {
        return false;
    }

    if (Wire.requestFrom((uint8_t)MPU_ADDR, (size_t)14, (bool)true) != 14) {
        return false;
    }

    int16_t rawAx = (Wire.read() << 8) | Wire.read();
    int16_t rawAy = (Wire.read() << 8) | Wire.read();
    int16_t rawAz = (Wire.read() << 8) | Wire.read();
    int16_t rawT  = (Wire.read() << 8) | Wire.read(); // Temperature (unused)
    (void)rawT;
    int16_t rawGx = (Wire.read() << 8) | Wire.read();
    int16_t rawGy = (Wire.read() << 8) | Wire.read();
    int16_t rawGz = (Wire.read() << 8) | Wire.read();

    s.ax = (float)rawAx / ACCEL_SCALE_8G;
    s.ay = (float)rawAy / ACCEL_SCALE_8G;
    s.az = (float)rawAz / ACCEL_SCALE_8G;

    s.gx = (float)rawGx / GYRO_SCALE_2000;
    s.gy = (float)rawGy / GYRO_SCALE_2000;
    s.gz = (float)rawGz / GYRO_SCALE_2000;

    return true;
}

// --- Compute Vector Magnitude |a| ---
float computeSVM(float ax, float ay, float az) {
    return sqrtf(ax * ax + ay * ay + az * az);
}

// --- Fast Edge Posture Change Check (Delta Theta) ---
float computeTiltDeg(int k) {
    // Pre-impact gravity vector: mean of [-50, -25]
    // Post-impact gravity vector: mean of [+15, +65]
    float p[3] = {0, 0, 0};
    float q[3] = {0, 0, 0};
    int np = 0, nq = 0;

    for (int offset = -50; offset <= -25; offset++) {
        int idx = (k + offset + TOTAL_SAMPLES) % TOTAL_SAMPLES;
        p[0] += ringBuffer[idx].ax;
        p[1] += ringBuffer[idx].ay;
        p[2] += ringBuffer[idx].az;
        np++;
    }

    for (int offset = 15; offset <= 65; offset++) {
        int idx = (k + offset + TOTAL_SAMPLES) % TOTAL_SAMPLES;
        q[0] += ringBuffer[idx].ax;
        q[1] += ringBuffer[idx].ay;
        q[2] += ringBuffer[idx].az;
        nq++;
    }

    float dot = 0, np2 = 0, nq2 = 0;
    for (int c = 0; c < 3; c++) {
        p[c] /= np;
        q[c] /= nq;
        dot += p[c] * q[c];
        np2 += p[c] * p[c];
        nq2 += q[c] * q[c];
    }

    float norm = sqrtf(np2) * sqrtf(nq2) + 1e-9f;
    float cs = dot / norm;
    if (cs > 1.0f) cs = 1.0f;
    if (cs < -1.0f) cs = -1.0f;

    return acosf(cs) * 57.2957795f;
}

// --- Transmit 150-sample Event Window to Host ---
void transmitBurstWindow() {
    Serial.println("=== EVENT_BURST_START ===");
    // Window starts at (bufHead) because ring buffer has exactly 150 elements
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        int idx = (bufHead + i) % TOTAL_SAMPLES;
        ImuSample &s = ringBuffer[idx];
        Serial.print(s.t_ms); Serial.print(",");
        Serial.print(s.ax, 4); Serial.print(",");
        Serial.print(s.ay, 4); Serial.print(",");
        Serial.print(s.az, 4); Serial.print(",");
        Serial.print(s.gx, 2); Serial.print(",");
        Serial.print(s.gy, 2); Serial.print(",");
        Serial.println(s.gz, 2);
    }
    Serial.println("=== EVENT_BURST_END ===");
}
