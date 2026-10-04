# fall_detector – bộ phát hiện ngã từ IMU 50 Hz

Triển khai của nghiên cứu `research_v10/` (xem `PLAN.md`, `RESULTS.md`). Chỉ phụ thuộc: numpy, pandas, scipy, scikit-learn, joblib.

## Mới: TinyFallNet chạy trên ESP32-S3 (1D-CNN, 87k MAC, 37,5 KB)
- `tinyfallnet/`: model, kết quả benchmark trên 5 metric, ngưỡng, script train và xuất C.
- `esp32_tinyfallnet/`: firmware PlatformIO đầy đủ (cảm biến + BLE + pulse + TinyFallNet chạy trên mạch).
- Trên dữ liệu của mình (5-fold chia theo người): F1 93,5, bắt ngã 65/67, nằm xuống không báo nhầm lần nào (0/10), 1,2 báo giả/giờ. RF 43 đặc trưng cùng giao thức đạt F1 91,0 và báo nhầm 5/10 lần nằm xuống.

## Cách hoạt động
```
mẫu IMU (t_ms, acc[3], gyr[3]) -> xử lý mất gói (≤0,4 s nội suy; dài hơn thì reset)
 -> kích hoạt: |acc| ≥ 1,6 g và là cực đại trong ±49 mẫu
 -> cửa sổ [-1,0 s ; +2,0 s] quanh đỉnh  (cảnh báo phát ra ~2,0 s sau va chạm)
 -> trích 43 đặc trưng bất biến hướng gắn (hướng/gravity, pha trước-sau, va chạm)
 -> cổng vật lý: đổi tư thế ≥ 5°  (mô hình chỉ chạy trên ~51% sự kiện)
 -> mô hình: RandomForest + HistGB (trung bình xác suất), ngưỡng 0,053 (≈2 báo giả/giờ trên dữ liệu normal)
 -> luật vật lý: đổi tư thế ≥ 40°
 -> mức cảnh báo:  FALL = mô hình ∧ luật   |   SUSPECTED = chỉ mô hình
```

## Dùng
```python
from fall_detector import FallDetector
det = FallDetector(mode="tiered")            # "tiered" (mặc định) | "model" | "rule"
for t_ms, acc, gyr in stream:                # stream thời gian thực
    for alert in det.push(t_ms, acc, gyr):   # có thể trả 0 hoặc 1 cảnh báo
        print(alert.tier, alert.t_impact_ms, alert.score, alert.tilt_deg)
alerts = det.flush()                          # cuối bản ghi: xử lý va chạm gần cuối
# hoặc cho cả file:  det.process_csv("file.csv")
```
Kiểm chứng phát lại: `python -m fall_detector.replay_eval` (chạy từ thư mục `IMU_data`).

## Bản rút gọn cho ESP32 (không cần ML)
`rule_params.json` và `fall_rule_params.h` chứa luật vật lý: sau đỉnh ≥1,6 g (cực đại trong ±49 mẫu), tính góc giữa gia tốc trung bình `[k-50,k-26]` và `[k+15,k+64]`; ≥ 40° → FALL (báo được sau 1,3 s). Bản tham chiếu Python khớp đặc trưng huấn luyện với sai số 0° trên 122 sự kiện (`python -m fall_detector.export_rule`).

## Triển khai Realtime: Kiến trúc Hybrid Edge-to-Host (Cách 3)
Hệ thống kết hợp tối ưu giữa biên phần cứng và trí tuệ nhân tạo:
1. **Phần cứng ESP32 (`esp32_firmware/esp32_fall_detector_node.ino`)**:
   - Thu thập IMU 50 Hz qua giao tiếp I2C không phụ thuộc thư viện ngoài.
   - Quản lý bộ đệm xoay vòng 150 mẫu (3.0s: [-1.0s, +2.0s]).
   - Kích hoạt ngưỡng xung chấn $|a| \ge 1.6g$, lọc góc nghiêng sơ bộ $\ge 25^\circ$.
   - Hai chế độ linh hoạt: Chế độ truyền trực tiếp 50Hz (STREAM) và Chế độ xung kích tiết kiệm pin (BURST).

2. **Máy chủ / Gateway (`realtime_engine.py`)**:
   - Tiếp nhận luồng Serial USB/BLE từ ESP32 hoặc chạy giả lập thời gian thực từ file CSV (`--replay`).
   - Kiểm định 2 tầng: Mô hình học máy Ensemble (RF + HistGB) $\wedge$ Luật vật lý $\Delta\theta \ge 40^\circ$.
   - **Quy trình an toàn**: Kích hoạt đếm ngược 15 giây âm thanh/hình ảnh. Cho phép người dùng bấm phím 'C' hoặc Spacebar (hoặc nút bấm trên thiết bị) để HỦY báo động giả. Hết 15 giây sẽ tự động kích hoạt Cứu hộ khẩn cấp SOS (`alerts_history.csv`).

```bash
# Chạy giả lập 1 ca ngã thật với tốc độ 50Hz:
python -m fall_detector.realtime_engine --replay danh_gia/fall_data_crop5s/person1/walking_1002_01.csv

# Chạy kết nối phần cứng ESP32 thật qua cổng COM:
python -m fall_detector.realtime_engine --port COM3 --baud 115200
```

## Kết quả (xem `research_v10/RESULTS.md` để biết đầy đủ)
| Chỉ số | Giá trị | Ghi chú |
|---|---|---|
| Recall ngã, mô hình ∧ luật (OOF, bỏ-lần-lượt-từng-người, 5 người, 65 ngã) | 96,9% | đo trên người chưa thấy |
| Báo giả trên hoạt động bình thường | ~1,0/giờ (OOF) | đo trên người chưa thấy |
| Độ trễ cảnh báo | ~2,0 s sau va chạm | cửa sổ cần phần "sau" |
| false_alarm (64 file, test cuối) | mô hình đơn 12/64 báo nhầm; luật nghiêng≥40° 3/64 | kết quả test đúng một lần; chế độ 2 mức được chọn *sau* khi thấy kết quả này |

## Giới hạn quan trọng
1. **Chỉ 5 người ngã, cùng một phiên, tất cả ngã kiểu "nằm ngang rồi nằm yên".** Ngã nhẹ, ngã có đứng dậy ngay, ngã cầu thang kiểu cũ (độ nghiêng 20–30°) sẽ bị bỏ sót. Recall chỉ có giá trị cho kiểu ngã này.
2. **Báo động giả khó** (ngồi xuống nhanh, nhấc vật nặng, trượt chân) vẫn lọt ở mức SUSPECTED; chỉ mức FALL (cần cả luật nghiêng ≥ 40°) mới đáng dùng để báo động tự động.
3. Mô hình ML chỉ cứu thêm 1 ca ngã so với luật vật lý thuần, đổi lại nhiều báo nhầm hơn trên ca khó; luật vật lý đơn giản là bản dự phòng an toàn.
4. Ngã gần đầu bản ghi (< 1 s ngữ cảnh) hoặc mất gói > 0,4 s quanh va chạm sẽ không được đánh giá.
5. Gia tốc bão hòa ~3,45 g; mô hình được huấn luyện với tần số lấy mẫu thật 50 Hz.

## Cấu trúc
`detector.py` (lõi), `features.py` (bộ đặc trưng, bản sao của `classifier_v5/feature_bank.py`), `artifacts/` (model.joblib, config.json), `replay_eval.py`, `export_rule.py`, `rule_params.json`, `fall_rule_params.h`.
