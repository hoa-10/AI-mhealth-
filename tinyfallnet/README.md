# TinyFallNet — 1D-CNN phát hiện ngã chạy trên ESP32-S3

Thay "43 đặc trưng + RandomForest" bằng một CNN nhỏ: **8,5k tham số, 87k MAC, 37,5 KB flash, 12 KB RAM**.
Firmware dùng model này nằm ở `../esp32_tinyfallnet/`.

## Đầu vào
- **Ứng viên:** đỉnh |a| ≥ 1,6 g, lớn nhất trong ±1 s.
- **Cửa sổ:** 150 mẫu = [−1 s, +2 s] ở 50 Hz, gồm gia tốc (g) và gyro (°/s).
- **Cắt gia tốc ±2 g mỗi trục:** cảm biến của mình là ADXL345 thang ±2 g, nên dữ liệu công khai cũng được cắt như vậy khi train.
- **6 kênh không phụ thuộc cách đeo**, tính theo hướng trọng lực trước va chạm:
  - |a|;
  - gia tốc dọc;
  - gia tốc ngang;
  - cos(nghiêng);
  - |gyro quanh trục dọc|;
  - |gyro quanh trục ngang|.

## Kiến trúc
conv(6→16, k5, s2) → 3 khối depthwise-separable (16→24→32→32, stride 2) → flatten 32×10 → FC16 → FC1 → sigmoid.

Dùng flatten thay vì pooling để model biết sự kiện xảy ra trước hay sau va chạm, vì cửa sổ luôn căn theo đỉnh va chạm.

## Train
- **Dữ liệu train:** UNIVRFall + SisFall (cắt về ±2 g), cộng với dữ liệu của mình (danh_gia) lặp ×5.
- **Ngưỡng:** 0,8625, ứng với 1 báo giả/giờ, lấy từ điểm out-of-fold chia theo người trên dữ liệu của mình.

## Kết quả (`results_own.csv`, `results_all_5metrics.csv`)

Dữ liệu của mình, 5-fold chia theo người, ngưỡng 1 báo giả/giờ chọn trong phần train:

| Model | Acc | Prec | Recall | F1 | AUC | Bắt ngã | Nằm xuống bị báo nhầm | Báo giả/giờ |
|---|---|---|---|---|---|---|---|---|
| **TinyFallNet (public + own)** | 97,98 | 90,28 | 97,01 | **93,53** | 99,67 | 65/67 | **0/10** | 1,20 |
| RF 43 đặc trưng | 97,08 | 84,62 | 98,51 | 91,03 | 99,76 | 66/67 | 5/10 | 1,20 |

Test cuối (một lần) trên `false_alarm` (64 kịch bản khó): báo nhầm **4/64** file.

Trên UNIVRFall và SisFall (5-fold chia theo người), mọi model học được đều ngang nhau ở F1 97–98. Luật web chỉ đạt F1 khoảng 88 ở đây.

## Giới hạn
- 67 cú ngã từ 5 người. Dữ liệu nằm xuống chỉ có 10 file từ 1 buổi thu.
- Chưa đo thời gian chạy trên mạch thật; firmware tự in ra khi khởi động.
- Các script train (`data.py`, `train_eval.py`, `own_eval.py`, `final.py`) chạy trong cấu trúc thư mục `IMU_data/src` ở máy local (cần `imu_classifier/benchmark/datasets.py` và `fall_detector_logic/features.py`). Ở đây chúng được lưu để tham khảo.
- `export_c.py` xuất `model.pt` ra `fall_cnn_weights.h`, và kiểm tra bản mô phỏng C khớp PyTorch.
