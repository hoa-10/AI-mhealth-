#include <Arduino.h>
#include <Wire.h>
#include <NimBLEDevice.h>
#include "fall_cnn.h"

#define SDA_PIN 3
#define SCL_PIN 4

#define ADXL345_ADDR 0x53
#define GYRO_ADDR    0x68
#define QMC_MAG_ADDR 0x0C

// Pulse Sensor (analog, PPG quang hoc) - chan ADC1 con trong, khong dung chung voi I2C
#define PULSE_PIN 5

#define SERVICE_UUID         "19b10000-e8f2-537e-4f6c-d104768a1214"
#define CHARACTERISTIC_UUID  "19b10002-e8f2-537e-4f6c-d104768a1214"
#define CHAR_FALL_ALERT_UUID "19b10003-e8f2-537e-4f6c-d104768a1214"

NimBLEServer* pServer = nullptr;
NimBLECharacteristic* pCharacteristic = nullptr;
NimBLECharacteristic* pFallAlertChar = nullptr;
bool deviceConnected = false;

// 12 x int16 Little-Endian = 24 bytes (goi tin stream lien tuc 50Hz)
// [0-2] Accel g*1000 | [3-5] Gyro dps*10 | [6-8] Mag uT*10
// [9] Pulse raw ADC (0-4095) | [10] Quick BPM (uoc tinh nhanh de xem live,
// khong phai so chinh thuc - nhip tim chuan se duoc xu ly o ngoai, VD qua LLM)
// [11] IBI ms (khoang cach nhip gan nhat, tinh nhanh on-device)
int16_t packet[12] = {0};

float ax = 0.0f, ay = 0.0f, az = 0.0f;
float gx = 0.0f, gy = 0.0f, gz = 0.0f;
float mx = 0.0f, my = 0.0f, mz = 0.0f;

float gx_offset = 0.0f, gy_offset = 0.0f, gz_offset = 0.0f;

// ===================== PULSE SENSOR (Quick Live Estimate) =====================
// Thuat toan bat dinh nguong dong (adaptive threshold), gion, chi de hien thi
// realtime cho nguoi dung biet dang do duoc nhip hay khong. Du lieu thô
// (pulse_raw) van duoc gui kem trong goi BLE de xu ly nhip tim chuan o ngoai.
int pulse_filter_buf[4] = {2000, 2000, 2000, 2000};
int pulse_filter_idx = 0;

float pulse_dc_baseline = 2000.0f;
float pulse_ac_signal = 0.0f;
float pulse_peak_amp = 15.0f;
float pulse_dyn_thresh = 5.0f;

volatile int pulseQuickBPM = 0;
volatile int pulseIBI = 750;
unsigned long pulseLastBeatTime = 0;
bool pulseDetected = false;
int pulse_rate_history[8] = {750, 750, 750, 750, 750, 750, 750, 750};
int pulse_rate_idx = 0;
int pulseRawLatest = 0;

class ServerCallbacks: public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, ble_gap_conn_desc* desc) {
        deviceConnected = true;
        Serial.println("[BLE] Client Connected!");
        // Thiet lap thong so ket noi toi uu cho iOS / BLE:
        // 16 * 1.25ms = 20ms interval (chuan 50Hz), timeout = 400 * 10ms = 4000ms (tranh ngat dot ngot)
        pServer->updateConnParams(desc->conn_handle, 16, 20, 0, 400);
    };
    void onDisconnect(NimBLEServer* pServer) {
        deviceConnected = false;
        Serial.println("[BLE] Client Disconnected, restarting advertising...");
        NimBLEDevice::startAdvertising();
    }
};

void writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

void initSensors() {
  Serial.println("[INIT] Khoi tao module 10DOF tren GPIO 3 (SDA) & GPIO 4 (SCL)...");

  // 1. ADXL345 (Gia toc)
  Wire.beginTransmission(ADXL345_ADDR);
  if (Wire.endTransmission(true) == 0) {
    writeReg(ADXL345_ADDR, 0x2D, 0x08);
    writeReg(ADXL345_ADDR, 0x31, 0x08);
    Serial.println("  [+] ADXL345 OK (0x53)");
  }

  // 2. Gyroscope ITG3200
  Wire.beginTransmission(GYRO_ADDR);
  if (Wire.endTransmission(true) == 0) {
    writeReg(GYRO_ADDR, 0x6B, 0x00);
    writeReg(GYRO_ADDR, 0x3E, 0x00);
    writeReg(GYRO_ADDR, 0x16, 0x18);
    writeReg(GYRO_ADDR, 0x6A, 0x00);
    writeReg(GYRO_ADDR, 0x37, 0x02);
    Serial.println("  [+] ITG3200 OK (0x68)");
  }

  // 3. QMC5883L (Tu ke)
  Wire.beginTransmission(QMC_MAG_ADDR);
  if (Wire.endTransmission(true) == 0) {
    writeReg(QMC_MAG_ADDR, 0x0B, 0x01);
    writeReg(QMC_MAG_ADDR, 0x09, 0x1D);
    Serial.println("  [+] QMC5883L OK (0x0C)");
  }

  // 4. Pulse Sensor (analog PPG)
  // - Dung ADC_6db (thay vi 11db) de dai do vua khop hon voi muc tin hieu
  //   thuc te (~1.5V), giam do phan giai mV/ma bi lang phi.
  // - Dung analogReadMilliVolts() (co hieu chinh eFuse) thay vi analogRead()
  //   tho, giam meo phi tuyen cua SAR-ADC ESP32 da ghi nhan tren forum.
  Serial.println("[INIT] Khoi tao Pulse Sensor tren GPIO 5 (ADC, 6dB, calibrated mV)...");
  analogReadResolution(12);
  analogSetPinAttenuation(PULSE_PIN, ADC_6db);

  long sum = 0;
  for (int i = 0; i < 50; i++) {
    sum += analogReadMilliVolts(PULSE_PIN);
    delay(5);
  }
  pulse_dc_baseline = (float)sum / 50.0f;
  for (int i = 0; i < 4; i++) pulse_filter_buf[i] = (int)pulse_dc_baseline;
  pulseLastBeatTime = millis();
  Serial.println("  [+] Pulse Sensor OK (GPIO 5)");
}

void readSensors() {
  // 1. Gia toc ADXL345
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x32);
  if (Wire.endTransmission(true) == 0) {
    if (Wire.requestFrom((int)ADXL345_ADDR, 6) == 6) {
      int16_t rx = Wire.read() | (Wire.read() << 8);
      int16_t ry = Wire.read() | (Wire.read() << 8);
      int16_t rz = Wire.read() | (Wire.read() << 8);
      ax = (float)rx * 0.0039f;
      ay = (float)ry * 0.0039f;
      az = (float)rz * 0.0039f;
    }
  }

  // 2. Con quay Gyro ITG3200
  Wire.beginTransmission(GYRO_ADDR);
  Wire.write(0x1D);
  if (Wire.endTransmission(true) == 0) {
    if (Wire.requestFrom((int)GYRO_ADDR, 6) == 6) {
      int16_t rx = (Wire.read() << 8) | Wire.read();
      int16_t ry = (Wire.read() << 8) | Wire.read();
      int16_t rz = (Wire.read() << 8) | Wire.read();
      gx = ((float)rx / 14.375f) - gx_offset;
      gy = ((float)ry / 14.375f) - gy_offset;
      gz = ((float)rz / 14.375f) - gz_offset;
    }
  }

  // 3. Tu ke QMC5883L
  Wire.beginTransmission(QMC_MAG_ADDR);
  Wire.write(0x0A);
  Wire.write(0x01);
  Wire.endTransmission(true);

  Wire.beginTransmission(QMC_MAG_ADDR);
  Wire.write(0x00);
  if (Wire.endTransmission(true) == 0) {
    if (Wire.requestFrom((int)QMC_MAG_ADDR, 6) == 6) {
      int16_t rx = Wire.read() | (Wire.read() << 8);
      int16_t ry = Wire.read() | (Wire.read() << 8);
      int16_t rz = Wire.read() | (Wire.read() << 8);
      mx = (float)rx / 30.0f;
      my = (float)ry / 30.0f;
      mz = (float)rz / 30.0f;
    }
  }
}

// Doc ADC nhieu lan lien tiep roi lay trung binh (oversampling) de giam
// nhieu ngau nhien cua SAR-ADC ESP32. Dung analogReadMilliVolts() (co hieu
// chinh eFuse rieng cho tung chip) thay vi analogRead() tho de giam meo
// phi tuyen - ca hai la khuyen nghi tu forum ESP32 cho tin hieu bien do nho.
int readPulseOversampled(int n) {
  long sum = 0;
  for (int i = 0; i < n; i++) {
    sum += analogReadMilliVolts(PULSE_PIN);
  }
  return (int)(sum / n);
}

// Doc + xu ly nhanh 1 mau Pulse Sensor: loc trung binh dong, tach baseline DC,
// bat dinh bang nguong dong (adaptive threshold) -> chi phuc vu hien thi live.
void readPulseQuick() {
  // Giam oversampling tu 64 xuong 8 mau: tiet kiem ~11ms CPU moi chu ky 20ms,
  // giup CPU khong bi qua tai va NimBLE BLE stack khong bi drop goi / mat ket noi.
  int raw = readPulseOversampled(8);
  pulseRawLatest = raw;

  // 1. Loc trung binh dong 4 mau
  pulse_filter_buf[pulse_filter_idx] = raw;
  pulse_filter_idx = (pulse_filter_idx + 1) % 4;
  float smoothed = (pulse_filter_buf[0] + pulse_filter_buf[1] + pulse_filter_buf[2] + pulse_filter_buf[3]) / 4.0f;

  // 2. Tach baseline DC troi cham
  pulse_dc_baseline += 0.015f * (smoothed - pulse_dc_baseline);

  // 3. Tin hieu song xung AC
  pulse_ac_signal = smoothed - pulse_dc_baseline;

  unsigned long now = millis();
  unsigned long timeSinceLast = now - pulseLastBeatTime;

  // 4. Cap nhat nguong dong do nhay cao (Min threshold = 4.0)
  if (pulse_ac_signal > pulse_dyn_thresh) {
    if (pulse_ac_signal > pulse_peak_amp) {
      pulse_peak_amp = pulse_ac_signal;
    }
  } else {
    pulse_peak_amp *= 0.997f; // Suy hao nhe
    if (pulse_peak_amp < 8.0f) pulse_peak_amp = 8.0f;
  }
  pulse_dyn_thresh = pulse_peak_amp * 0.45f;
  if (pulse_dyn_thresh < 3.5f) pulse_dyn_thresh = 3.5f;

  // 5. Bat dinh nhip tim (300ms - 1500ms ~ 40-200 BPM)
  if (timeSinceLast > 320) {
    if ((pulse_ac_signal > pulse_dyn_thresh) && !pulseDetected && (timeSinceLast > (unsigned long)(pulseIBI * 0.55f))) {
      pulseDetected = true;
      pulseIBI = timeSinceLast;
      pulseLastBeatTime = now;

      pulse_rate_history[pulse_rate_idx] = pulseIBI;
      pulse_rate_idx = (pulse_rate_idx + 1) % 8;

      long sumIBI = 0;
      for (int i = 0; i < 8; i++) sumIBI += pulse_rate_history[i];
      int avgIBI = sumIBI / 8;

      if (avgIBI > 0) {
        int calcBPM = 60000 / avgIBI;
        if (calcBPM >= 45 && calcBPM <= 180) {
          pulseQuickBPM = calcBPM;
        }
      }
    }
  }

  if (pulse_ac_signal < (pulse_dyn_thresh * 0.2f) && pulseDetected) {
    pulseDetected = false;
  }

  if (timeSinceLast > 2500) {
    pulseQuickBPM = 0;
  }
}

// ===================== TinyFallNet FALL DETECTOR (on-device) =====================
// Ung vien c = mau co |a| >= 1.6 g va la dinh lon nhat trong [c-1 s, c+1 s] (giong find_peaks luc train).
// Khi da co du mau c+99: cua so 150 mau [c-50, c+100) -> TinyFallNet -> xac suat nga.
// Nhieu ung vien co the cho cung luc (hang doi), nen buoc chan manh ngay truoc cu nga khong che mat cu nga.
// FALL neu xac suat >= FALL_CNN_THRESHOLD (0.8625, chon o muc 1 bao gia/gio tren du lieu cua minh).
// FALL_REQUIRE_TILT = 1: chi bao do khi them goc nghieng >= 40 do (che do 2 muc; CHUA danh gia, de mac dinh 0).
#define FALL_REQUIRE_TILT   0
#define FALL_TILT_DEG       40.0f
#define FALL_TRIGGER_G      1.6f
#define FALL_HALF           50      // dinh phai lon nhat trong +-1 s
#define FALL_ALERT_MS       15000
#define RING                200     // can lui toi da 149 mau
#define MAX_PENDING         4

static float ring_acc[RING][3], ring_gyr[RING][3], ring_svm[RING];
static uint32_t n_samples = 0;
static uint32_t pending[MAX_PENDING];
static int n_pending = 0;
static float cnn_win[FALL_CNN_WIN][6];
static bool fall_alert_active = false;
static unsigned long fall_alert_start = 0;

static float window_tilt_deg() {               // goc giua trong luc [0,35) va [100,150) cua cua so
  float p[3] = {0, 0, 0}, q[3] = {0, 0, 0};
  for (int t = 0; t < 35; t++) for (int j = 0; j < 3; j++) p[j] += cnn_win[t][j];
  for (int t = 100; t < FALL_CNN_WIN; t++) for (int j = 0; j < 3; j++) q[j] += cnn_win[t][j];
  float d = p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  float c = d / (sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) * sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) + 1e-9f);
  c = c > 1.f ? 1.f : (c < -1.f ? -1.f : c);
  return acosf(c) * 57.2957795f;
}

static void send_fall_status(const char* msg) {
  if (deviceConnected && pFallAlertChar != nullptr) {
    pFallAlertChar->setValue((uint8_t*)msg, strlen(msg));
    pFallAlertChar->notify();
  }
}

static void run_fall_model(uint32_t c) {
  // copy [peak-50, peak+100) tu ring buffer, cat gia toc ve +-2 g nhu luc train
  for (int t = 0; t < FALL_CNN_WIN; t++) {
    uint32_t i = (c - 50 + t) % RING;
    for (int j = 0; j < 3; j++) {
      float a = ring_acc[i][j];
      cnn_win[t][j] = a > 2.f ? 2.f : (a < -2.f ? -2.f : a);
      cnn_win[t][3 + j] = ring_gyr[i][j];
    }
  }
  uint32_t t0 = micros();
  float p = fall_cnn_score(cnn_win);
  uint32_t dt = micros() - t0;
  float tilt = window_tilt_deg();
  bool fall = p >= fall_cnn_threshold() && (!FALL_REQUIRE_TILT || tilt >= FALL_TILT_DEG);

  Serial.printf("[TinyFallNet] dinh %.2fg | P_fall %.1f%% (nguong %.1f%%) | nghieng %.1f deg | suy luan %lu us\n",
                ring_svm[c % RING], p * 100.f, fall_cnn_threshold() * 100.f, tilt, (unsigned long)dt);
  if (fall) {
    fall_alert_active = true; fall_alert_start = millis();
    Serial.println(">>> [SOS] TinyFallNet XAC NHAN NGA <<<");
    char buf[32]; snprintf(buf, sizeof(buf), "FALL:%.0f%%:%.1fdeg", p * 100.f, tilt);
    send_fall_status(buf);
  } else if (p >= fall_cnn_threshold()) {
    Serial.println("[TinyFallNet] Model nghi nga nhung goc nghieng < 40 deg -> khong bao dong");
  }
}

// goi moi mau (50 Hz) sau readSensors()
static void fall_detector_step() {
  uint32_t k = n_samples++, i = k % RING;
  ring_acc[i][0] = ax; ring_acc[i][1] = ay; ring_acc[i][2] = az;
  ring_gyr[i][0] = gx; ring_gyr[i][1] = gy; ring_gyr[i][2] = gz;
  float cx = fminf(fmaxf(ax, -2.f), 2.f), cy = fminf(fmaxf(ay, -2.f), 2.f), cz = fminf(fmaxf(az, -2.f), 2.f);
  ring_svm[i] = sqrtf(cx * cx + cy * cy + cz * cz);

  // 1) mau c = k-50 co phai ung vien? (|a| >= 1.6 g, lon nhat trong [c-50, c+50], gap dau tien neu bang nhau)
  if (k >= 2 * FALL_HALF) {
    uint32_t c = k - FALL_HALF; float sc = ring_svm[c % RING];
    if (sc >= FALL_TRIGGER_G) {
      bool is_peak = true;
      for (uint32_t j = c - FALL_HALF; j <= k && is_peak; j++) {
        float sj = ring_svm[j % RING];
        if (sj > sc || (sj == sc && j < c)) is_peak = false;
      }
      if (is_peak && n_pending < MAX_PENDING) pending[n_pending++] = c;
    }
  }
  // 2) ung vien nao da du 2 s sau dinh -> chay model
  for (int q = 0; q < n_pending; q++) {
    if (k == pending[q] + 99) {
      run_fall_model(pending[q]);
      pending[q] = pending[--n_pending]; q--;
    }
  }

  if (fall_alert_active && millis() - fall_alert_start >= FALL_ALERT_MS) {
    fall_alert_active = false;
    send_fall_status("STATUS:OK");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[SYSTEM] ESP32-S3 10DOF + Pulse Sensor + TinyFallNet (1D-CNN on-device)");

  Wire.begin(SDA_PIN, SCL_PIN, 400000);
  Wire.setTimeOut(10);
  initSensors();

  // Kiem tra model: sai so voi PyTorch tren cua so mau + do thoi gian suy luan
  Serial.printf("[TinyFallNet] self-test |err| = %.2e (phai < 1e-4)\n", fall_cnn_selftest());
  uint32_t t0 = micros();
  for (int r = 0; r < 20; r++) fall_cnn_selftest();
  Serial.printf("[TinyFallNet] thoi gian suy luan: %.3f ms (TB 20 lan, CPU %d MHz)\n", (micros() - t0) / 20000.0f, getCpuFrequencyMhz());

  NimBLEDevice::init("ESP32_IMU");
  NimBLEDevice::setPower(ESP_PWR_LVL_P9);
  pServer = NimBLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());
  NimBLEService* pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(CHARACTERISTIC_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
  pCharacteristic->setValue((uint8_t*)packet, sizeof(packet));
  // Bao dong nga tu ESP32: "FALL:<xac suat>%:<goc>deg" hoac "STATUS:OK"
  pFallAlertChar = pService->createCharacteristic(CHAR_FALL_ALERT_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
  pFallAlertChar->setValue((uint8_t*)"STATUS:OK", 9);
  pService->start();

  NimBLEAdvertising* pAdvertising = NimBLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setName("ESP32_IMU");
  pAdvertising->setScanResponse(true);
  pAdvertising->start();
  Serial.println("[BLE] Dang phat song Bluetooth 'ESP32_IMU' - SAN SANG!");
}

unsigned long lastTime = 0;
unsigned long lastPrint = 0;

void loop() {
  unsigned long now = millis();
  if (now - lastTime >= 20) {   // 50 Hz
    lastTime = now;
    readSensors();
    readPulseQuick();
    fall_detector_step();

    // 24 byte: 9 x IMU + pulse raw + BPM + IBI (-999 khi dang bao dong nga)
    packet[0] = (int16_t)(ax * 1000.0f);
    packet[1] = (int16_t)(ay * 1000.0f);
    packet[2] = (int16_t)(az * 1000.0f);
    packet[3] = (int16_t)(gx * 10.0f);
    packet[4] = (int16_t)(gy * 10.0f);
    packet[5] = (int16_t)(gz * 10.0f);
    packet[6] = (int16_t)(mx * 10.0f);
    packet[7] = (int16_t)(my * 10.0f);
    packet[8] = (int16_t)(mz * 10.0f);
    packet[9]  = (int16_t)pulseRawLatest;
    packet[10] = (int16_t)pulseQuickBPM;
    packet[11] = fall_alert_active ? (int16_t)-999 : (int16_t)pulseIBI;

    if (deviceConnected && pCharacteristic != nullptr) {
      pCharacteristic->setValue((uint8_t*)packet, sizeof(packet));
      pCharacteristic->notify();
    }

    if (now - lastPrint >= 500) {
      lastPrint = now;
      Serial.printf("ACC[%.2f, %.2f, %.2f] | GYR[%.1f, %.1f, %.1f] | FALL: %s | PULSE raw:%d BPM:%d | BLE: %s\n",
                    ax, ay, az, gx, gy, gz, fall_alert_active ? "!!! SOS !!!" : "MONITORING",
                    pulseRawLatest, pulseQuickBPM, deviceConnected ? "DA KET NOI" : "DANG CHO");
    }
  } else {
    delay(1);   // nhuong CPU cho BLE stack
  }
}
