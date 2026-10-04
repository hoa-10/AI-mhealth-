"""
Real-time Fall Detection Engine (Hybrid Edge-to-Host).
Connects to an ESP32 node via Serial (USB/COM) or simulates live streaming from CSV.

Usage:
  # 1. Live hardware mode (ESP32 connected via USB):
  python -m fall_detector.realtime_engine --port COM3 --baud 115200

  # 2. Interactive Simulator Mode (Replaying a real fall recording at 50 Hz):
  python -m fall_detector.realtime_engine --replay danh_gia/fall_data_crop5s/person1/fall_forward_01.csv

  # 3. Interactive Simulator Mode (Replaying a difficult false alarm at 50 Hz):
  python -m fall_detector.realtime_engine --replay danh_gia/false_alarm/FA_jump_bed_01.csv

Features:
  - 50 Hz real-time streaming parser with gap handling & burst window support.
  - Frozen Ensemble ML model + Physics Angle Rule verification.
  - 15-second audio/visual countdown with instant user cancellation (Press 'C' or Space).
  - Emergency dispatch logger upon countdown expiration.
"""

import os
import sys
import time
import argparse
import threading
import warnings
import struct
import asyncio
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Windows UTF-8 console output fix
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Audio support on Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Non-blocking keyboard support on Windows
try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

from fall_detector.detector import FallDetector, Alert

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts_history.csv")


class AudioAlerter:
    """Non-blocking audio alert player using Windows winsound."""
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None

    def start_countdown_beep(self):
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._beep_loop, daemon=True)
        self._thread.start()

    def _beep_loop(self):
        if not HAS_WINSOUND:
            return
        while not self._stop_event.is_set():
            try:
                # Warning beep: 880 Hz, 180 ms
                winsound.Beep(880, 180)
            except Exception:
                pass
            time.sleep(0.5)

    def play_emergency_sos(self):
        self.stop()
        if not HAS_WINSOUND:
            return
        def _sos():
            for freq in [1200, 1500, 1200, 1500]:
                try:
                    winsound.Beep(freq, 250)
                except Exception:
                    pass
        threading.Thread(target=_sos, daemon=True).start()

    def play_cancel_tone(self):
        self.stop()
        if not HAS_WINSOUND:
            return
        def _tone():
            try:
                winsound.Beep(523, 100) # C5
                winsound.Beep(659, 150) # E5
            except Exception:
                pass
        threading.Thread(target=_tone, daemon=True).start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)


class RealtimeFallEngine:
    def __init__(self, mode="tiered", fp_budget=2, countdown_sec=15):
        self.detector = FallDetector(mode=mode, fp_budget=fp_budget)
        self.alerter = AudioAlerter()
        self.countdown_sec = countdown_sec

        # State: 'NORMAL', 'COUNTDOWN', 'DISPATCHED'
        self.state = "NORMAL"
        self.active_alert = None
        self.countdown_start = 0.0
        self.last_countdown_display = -1

        self.sample_count = 0
        self.start_wall_time = time.time()
        self.running = True

        self._ensure_log_file()

    def _ensure_log_file(self):
        if not os.path.exists(LOG_FILE):
            cols = ["timestamp", "impact_t_ms", "tier", "score", "tilt_deg", "peak_g", "status"]
            pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

    def log_event(self, alert: Alert, status: str):
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "impact_t_ms": alert.t_impact_ms,
            "tier": alert.tier,
            "score": f"{alert.score:.4f}",
            "tilt_deg": f"{alert.tilt_deg:.1f}",
            "peak_g": f"{alert.peak_g:.2f}",
            "status": status
        }
        df = pd.DataFrame([record])
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

    def check_user_cancellation(self):
        """Check for non-blocking keypress to cancel alert."""
        if not HAS_MSVCRT:
            return False
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            # 'c', 'C', or Space (b' ') or Enter (b'\r')
            if ch in (b'c', b'C', b' ', b'\r'):
                return True
        return False

    def on_sample(self, t_ms: float, acc: list, gyr: list):
        self.sample_count += 1
        alerts = self.detector.push(t_ms, acc, gyr)
        for al in alerts:
            self._handle_alert(al)

        # Update countdown state machine
        if self.state == "COUNTDOWN":
            elapsed = time.time() - self.countdown_start
            remaining = int(np.ceil(self.countdown_sec - elapsed))

            # User cancellation check
            if self.check_user_cancellation():
                self.alerter.play_cancel_tone()
                print(f"\n>> [CANCELLED] Người dùng đã bấm HỦY báo động thành công! (False Alarm tránh được)")
                print(f">> Hệ thống trở lại trạng thái GIÁM SÁT BÌNH THƯỜNG.\n")
                self.log_event(self.active_alert, "CANCELLED_BY_USER")
                self.state = "NORMAL"
                self.active_alert = None
                return

            if remaining <= 0:
                self.alerter.play_emergency_sos()
                print(f"\n==================================================================")
                print(f"!!! [EMERGENCY DISPATCH] HẾT THỜI GIAN CHỜ (15s) KHÔNG CÓ PHẢN HỒI !!!")
                print(f"!!! XÁC NHẬN TÉ NGÃ NGUY HIỂM! ĐANG GỬI TÍN HIỆU CỨU HỘ SOS...")
                print(f"!!! Thời điểm va chạm: {self.active_alert.t_impact_ms:.0f} ms | Lực va chạm: {self.active_alert.peak_g:.2f}g | Góc nghiêng: {self.active_alert.tilt_deg:.1f}°")
                print(f"==================================================================\n")
                self.log_event(self.active_alert, "EMERGENCY_DISPATCHED")
                self.state = "DISPATCHED"
            elif remaining != self.last_countdown_display:
                self.last_countdown_display = remaining
                print(f"  [ĐANG ĐẾM NGƯỢC] Còn {remaining:2d} giây trước khi gửi SOS... (Nhấn 'C' hoặc Space để HỦY)")

    def _handle_alert(self, al: Alert):
        if al.tier == "FALL":
            print(f"\n" + "="*66)
            print(f"⚠️  [BÁO ĐỘNG CẤP 1 - PHÁT HIỆN TÉ NGÃ (FALL)]")
            print(f"   Thời điểm va chạm: {al.t_impact_ms:.0f} ms")
            print(f"   Lực đỉnh va chạm : {al.peak_g:.2f} g  (Ngưỡng: >= 1.6g)")
            print(f"   Góc nghiêng dáng : {al.tilt_deg:.1f} °  (Ngưỡng: >= 40.0°)")
            print(f"   Độ tự tin Model  : {al.score * 100:.1f} %")
            print(f"   Bắt đầu đếm ngược {self.countdown_sec}s để người dùng hủy nếu bấm nhầm...")
            print(f"   👉 Nhấn phím 'C' hoặc phím SPACE để HỦY BÁO ĐỘNG")
            print("="*66)

            self.active_alert = al
            self.state = "COUNTDOWN"
            self.countdown_start = time.time()
            self.last_countdown_display = self.countdown_sec
            self.alerter.start_countdown_beep()

        elif al.tier == "SUSPECTED":
            print(f"\n🔔 [CẢNH BÁO NHẸ - SUSPECTED]")
            print(f"   Phát hiện xung chấn {al.peak_g:.2f}g, góc nghiêng {al.tilt_deg:.1f}°, điểm số AI: {al.score:.3f}")
            print(f"   (Chưa đủ điều kiện ngã hẳn -> Tiếp tục theo dõi tư thế phục hồi)\n")


# ---------------- Stream Reader Modes ----------------

def run_serial_stream(engine: RealtimeFallEngine, port: str, baud: int):
    """Read from hardware ESP32 via Serial."""
    import serial
    print(f">> [SERIAL] Đang mở cổng {port} ở tốc độ {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(1.5) # Wait for ESP32 reboot
        print(f">> [SERIAL] Kết nối thành công! Đang nhận dữ liệu 50Hz...")
    except Exception as e:
        print(f"❌ [SERIAL ERROR] Không thể mở cổng {port}: {e}")
        return

    in_burst = False
    try:
        while engine.running:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue

            if line.startswith("#"):
                print(f"[ESP32 MSG] {line}")
                continue

            if "EVENT_BURST_START" in line:
                in_burst = True
                continue
            if "EVENT_BURST_END" in line:
                in_burst = False
                continue

            parts = line.split(",")
            if len(parts) >= 7:
                try:
                    t_ms = float(parts[0])
                    ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                    gx, gy, gz = float(parts[4]), float(parts[5]), float(parts[6])
                    engine.on_sample(t_ms, [ax, ay, az], [gx, gy, gz])
                except ValueError:
                    pass
    except KeyboardInterrupt:
        print("\n>> Dừng nhận dữ liệu theo yêu cầu người dùng.")
    finally:
        ser.close()
        engine.alerter.stop()


def run_csv_replay_stream(engine: RealtimeFallEngine, csv_path: str, speed_multiplier: float = 1.0):
    """Replay a recorded CSV file at exact 50Hz (20ms/sample) timing."""
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file CSV: {csv_path}")
        return

    print(f">> [SIMULATOR] Đang phát lại file: {csv_path}")
    print(f">> Tốc độ phát: {speed_multiplier}x (mỗi mẫu cách nhau {20.0 / speed_multiplier:.1f} ms)")
    print(f">> Bấm 'C' hoặc Space khi có còi báo động để kiểm tra tính năng Hủy Báo Động!\n")

    df = pd.read_csv(csv_path)
    t_vals = df["timestamp_ms"].values if "timestamp_ms" in df.columns else np.arange(len(df)) * 20.0
    acc = df[["accX_g", "accY_g", "accZ_g"]].values
    gyr = df[["gyrX_dps", "gyrY_dps", "gyrZ_dps"]].values

    n_samples = len(df)
    sample_dt = 0.020 / speed_multiplier

    t_prev = time.time()
    for i in range(n_samples):
        engine.on_sample(float(t_vals[i]), acc[i].tolist(), gyr[i].tolist())

        # Exact clock synchronization
        t_now = time.time()
        sleep_time = sample_dt - (t_now - t_prev)
        if sleep_time > 0:
            time.sleep(sleep_time)
        t_prev = time.time()

    # Flush detector remaining window
    flushed = engine.detector.flush()
    for al in flushed:
        engine._handle_alert(al)

    # Allow remaining countdown time to tick if still active
    while engine.state == "COUNTDOWN":
        time.sleep(0.2)
        engine.on_sample(float(t_vals[-1]), acc[-1].tolist(), gyr[-1].tolist())

    engine.alerter.stop()
    print(f"\n>> [SIMULATOR] Hoàn thành phát lại {n_samples} mẫu. Trạng thái kết thúc: {engine.state}")


# Bluetooth Low Energy (BLE) Configuration for 10DOF IMU Node
BLE_DEFAULT_NAME = "ESP32_IMU"
BLE_SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
BLE_CHAR_UUID    = "19b10002-e8f2-537e-4f6c-d104768a1214"


async def scan_ble_devices():
    """Scan and list nearby BLE devices, identifying any ESP32 IMU node."""
    try:
        from bleak import BleakScanner
        print(">> [BLE SCAN] Đang quét các thiết bị Bluetooth xung quanh trong 5 giây...")
        devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
        if not devices:
            print(">> [BLE SCAN] Không phát hiện thiết bị Bluetooth nào trong phạm vi.")
            return
        print(f">> [BLE SCAN] Phát hiện {len(devices)} thiết bị:")
        found_target = False
        for addr, (d, adv) in devices.items():
            name = d.name or adv.local_name or "(Không rõ tên)"
            is_imu = (BLE_DEFAULT_NAME.lower() in name.lower())
            mark = "🎯 [ESP32_IMU]" if is_imu else "  "
            rssi = adv.rssi if hasattr(adv, 'rssi') else "N/A"
            print(f" {mark} - {name:25s} | MAC: {d.address} | RSSI: {rssi} dBm")
            if is_imu:
                found_target = True
        if found_target:
            print("\n✅ Đã phát hiện thấy thiết bị ESP32_IMU! Bạn có thể kết nối ngay bằng lệnh:")
            print("   python -m fall_detector.realtime_engine --ble")
        else:
            print(f"\n💡 Chưa thấy thiết bị có tên '{BLE_DEFAULT_NAME}'. Hãy chắc chắn ESP32 đã bật nguồn.")
    except Exception as e:
        print(f"❌ [BLE SCAN ERROR] Lỗi khi quét Bluetooth: {e}")


async def _ble_stream_worker(engine: RealtimeFallEngine, device_name: str, service_uuid: str, char_uuid: str):
    from bleak import BleakScanner, BleakClient
    print(f">> [BLE] Đang tìm kiếm thiết bị Bluetooth '{device_name}' (Service: {service_uuid[:8]}...)...")

    def match_fn(d, adv):
        if d.name and device_name.lower() in d.name.lower():
            return True
        if adv.service_uuids and any(service_uuid.lower() in str(u).lower() for u in adv.service_uuids):
            return True
        return False

    device = await BleakScanner.find_device_by_filter(match_fn, timeout=10.0)
    if not device:
        print(f"❌ [BLE] Không tìm thấy thiết bị '{device_name}'.")
        print("   Gợi ý: Hãy chắc chắn ESP32 đã nạp code và bật nguồn.")
        print("   Kiểm tra danh sách Bluetooth: python -m fall_detector.realtime_engine --ble-scan")
        return

    print(f">> [BLE] Đã tìm thấy: {device.name} [{device.address}]")
    print(f">> [BLE] Đang thiết lập kết nối GATT...")

    packet_counter = 0

    def notification_handler(sender, data: bytearray):
        nonlocal packet_counter
        packet_counter += 1
        n_bytes = len(data)
        if n_bytes < 12:
            return

        # Unpack Little-Endian int16
        n_shorts = n_bytes // 2
        p = struct.unpack("<" + "h" * n_shorts, data[:n_shorts * 2])

        # 10DOF payload format (same as pulse_sensor/IMU_sensor_10DOF/firmware_platformio):
        # [0..2] ax, ay, az * 1000 (g)
        # [3..5] gx, gy, gz * 10 (dps)
        # [6..8] mx, my, mz * 10 (uT)
        # [9] pulseRaw, [10] bpm, [11] ibi
        ax = float(p[0]) / 1000.0
        ay = float(p[1]) / 1000.0
        az = float(p[2]) / 1000.0
        gx = float(p[3]) / 10.0
        gy = float(p[4]) / 10.0
        gz = float(p[5]) / 10.0

        bpm = p[10] if len(p) >= 11 else 0
        t_ms = time.time() * 1000.0

        engine.on_sample(t_ms, [ax, ay, az], [gx, gy, gz])

        if packet_counter % 50 == 0:
            svm = np.sqrt(ax**2 + ay**2 + az**2)
            sys.stdout.write(f"\r>> [BLE 50Hz] #{packet_counter:6d} | Acc:[{ax:+5.2f},{ay:+5.2f},{az:+5.2f}]g (|a|={svm:4.2f}g) | Gy:[{gx:+5.0f},{gy:+5.0f},{gz:+5.0f}] | BPM:{bpm:2d} | Trạng thái: {engine.state}    ")
            sys.stdout.flush()

    try:
        async with BleakClient(device) as client:
            print(f">> [BLE] Kết nối Bluetooth GATT thành công!")
            print(f">> [BLE] Đang đăng ký nhận thông báo từ Characteristic {char_uuid}...")
            await client.start_notify(char_uuid, notification_handler)
            print(f">> [BLE] ĐANG THU THẬP & GIÁM SÁT TÉ NGÃ 50Hz QUA BLUETOOTH!")
            print(f">> Khi có chuông báo té ngã, nhấn phím 'C' hoặc Space trên bàn phím để HỦY.")
            print(f">> Nhấn Ctrl+C để ngắt kết nối an toàn.\n")

            while engine.running and client.is_connected:
                # Check user cancellation during countdown
                if engine.state == "COUNTDOWN":
                    if engine.check_user_cancellation():
                        engine.alerter.play_cancel_tone()
                        print(f"\n>> [CANCELLED] Người dùng đã bấm HỦY báo động thành công!")
                        engine.log_event(engine.active_alert, "CANCELLED_BY_USER")
                        engine.state = "NORMAL"
                        engine.active_alert = None
                await asyncio.sleep(0.05)

            if not client.is_connected:
                print("\n⚠️ [BLE] Mất kết nối Bluetooth với ESP32.")
    except Exception as e:
        print(f"\n❌ [BLE ERROR] Lỗi trong phiên Bluetooth: {e}")
    finally:
        engine.alerter.stop()


def run_ble_stream(engine: RealtimeFallEngine, device_name: str = BLE_DEFAULT_NAME, service_uuid: str = BLE_SERVICE_UUID, char_uuid: str = BLE_CHAR_UUID):
    """Entry point for BLE streaming."""
    try:
        asyncio.run(_ble_stream_worker(engine, device_name, service_uuid, char_uuid))
    except KeyboardInterrupt:
        print("\n>> Đã dừng nhận dữ liệu Bluetooth theo yêu cầu.")
        engine.alerter.stop()


def list_serial_ports():
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print(">> Không tìm thấy cổng COM nào đang kết nối.")
        else:
            print(">> Các cổng COM khả dụng:")
            for p in ports:
                print(f"   - {p.device}: {p.description}")
    except Exception as e:
        print(f"Lỗi kiểm tra cổng: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Edge-to-Host Real-Time Fall Detector")
    parser.add_argument("--port", type=str, default=None, help="Serial COM port (e.g. COM3, COM4)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--ble", action="store_true", help="Connect to ESP32 IMU via Bluetooth Low Energy (BLE)")
    parser.add_argument("--ble-scan", action="store_true", help="Scan nearby Bluetooth BLE devices and exit")
    parser.add_argument("--ble-name", type=str, default=BLE_DEFAULT_NAME, help=f"BLE device name (default: {BLE_DEFAULT_NAME})")
    parser.add_argument("--replay", type=str, default=None, help="Path to CSV file to simulate real-time stream")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation playback speed multiplier (default: 1.0)")
    parser.add_argument("--mode", type=str, default="tiered", choices=["tiered", "model", "rule"])
    parser.add_argument("--countdown", type=int, default=15, help="Countdown window in seconds before SOS dispatch")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")

    args = parser.parse_args()

    if args.list_ports:
        list_serial_ports()
        return

    if args.ble_scan:
        asyncio.run(scan_ble_devices())
        return

    engine = RealtimeFallEngine(mode=args.mode, countdown_sec=args.countdown)

    if args.replay:
        run_csv_replay_stream(engine, args.replay, speed_multiplier=args.speed)
    elif args.ble:
        run_ble_stream(engine, device_name=args.ble_name)
    elif args.port:
        run_serial_stream(engine, args.port, args.baud)
    else:
        print(">> Bạn chưa chọn phương thức kết nối. Hãy chọn một trong các tùy chọn:")
        print("   1. Kết nối Bluetooth (ESP32 10DOF BLE):")
        print("      python -m fall_detector.realtime_engine --ble")
        print("   2. Quét tìm thiết bị Bluetooth:")
        print("      python -m fall_detector.realtime_engine --ble-scan")
        print("   3. Kết nối cổng Serial USB:")
        print("      python -m fall_detector.realtime_engine --port COM3")
        print("   4. Chạy giả lập phát lại file CSV tại 50Hz:")
        print("      python -m fall_detector.realtime_engine --replay danh_gia/fall_data_crop5s/person1/walking_1002_01.csv")


if __name__ == "__main__":
    main()
