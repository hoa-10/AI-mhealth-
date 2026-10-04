# Firmware ESP32-S3: phát hiện ngã bằng TinyFallNet ngay trên mạch

Phần cứng giữ nguyên như firmware `IMU_sensor_10DOF`:
- ADXL345 (±2 g), ITG3200, QMC5883L, I2C trên GPIO 3/4;
- Pulse sensor trên GPIO 5;
- BLE tên `ESP32_IMU`, gói 24 byte ở 50 Hz, giữ nguyên định dạng cũ.

## Build và nạp
```
pio run -t upload && pio device monitor
```
Lúc khởi động, Serial in ra:
- `self-test |err|`: phải < 1e-4, tức model trên mạch cho kết quả giống PyTorch;
- `thoi gian suy luan`: số ms thật mỗi lần chạy model.

## Cách hoạt động
1. Mỗi mẫu được lưu vào bộ đệm vòng 200 mẫu.
2. Mẫu c là ứng viên nếu |a| ≥ 1,6 g và là **đỉnh lớn nhất trong ±1 s**, giống `find_peaks` lúc train. Hàng đợi giữ tối đa 4 ứng viên, nên một bước chân mạnh ngay trước cú ngã không che mất cú ngã.
3. Khi đã có đủ 2 s sau đỉnh, cửa sổ [c−1 s, c+2 s] (gia tốc cắt ±2 g) được đưa vào `fall_cnn_score()`.
4. Nếu xác suất ≥ 0,8625:
   - BLE characteristic `19b10003-…` gửi `FALL:<xác suất>%:<góc>deg`;
   - `packet[11] = -999` trong 15 s;
   - sau 15 s gửi `STATUS:OK`.
5. `FALL_REQUIRE_TILT 1` bật thêm điều kiện góc nghiêng ≥ 40°. Chế độ này **chưa được đánh giá**, nên mặc định tắt.

Đã mô phỏng đúng cơ chế trigger này bằng Python (`../tinyfallnet/sim_firmware.py`) trên dữ liệu của mình: 67/67 ngã, 0/10 nằm xuống, 0 báo giả. Đây là số in-sample, vì model cuối đã train trên chính dữ liệu đó; nó chỉ xác nhận đường ống xử lý khớp với lúc train.

Build bằng PlatformIO: RAM 15,7%, flash 44,8%.

**Lưu ý:** trang web `fall_monitor.html` hiện chưa đọc characteristic báo ngã này, mà vẫn dùng luật riêng của nó.
