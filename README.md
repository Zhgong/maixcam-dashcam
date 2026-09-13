# 🚗 MaixCAM 2 AI Dashcam & FCW Guard

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Sipeed%20MaixCAM%202-orange.svg)](https://wiki.sipeed.com/maixcam)
[![NPU](https://img.shields.io/badge/NPU-1.0%20TOPS-green.svg)](https://wiki.sipeed.com/maixcam)
[![Tests](https://img.shields.io/badge/tests-51%2F51%20passing-brightgreen.svg)](tests/)

**English** | [中文文档 (Chinese)](README_ZH.md)

---

## 1. Vision & Core Philosophy

**MaixCAM 2 AI Dashcam & FCW Guard** is an edge-native smart dashcam and Forward Collision Warning (FCW) assistance system tailored specifically for the **Sipeed MaixCAM 2** hardware platform.

Leveraging the on-board **1.0 TOPS NPU**, **4K VPU hardware video pipeline**, and **LSM6DSOWTR 6-axis IMU**, this project delivers an ultra-low latency, zero-cloud-dependency ADAS assistant designed for geeks, road-trippers, and camper vans.

### Design Principles:
* **Zero Nuisance & Low False-Alarm Rate**: Tail-gating in stop-and-go traffic or waiting at red lights triggers **0% false alarms**; alerts fire strictly when relative approach speed ($v_{\text{rel}}$) and Time-To-Collision ($TTC$) breach safety thresholds.
* **100% Edge-Offline Loop**: Vision inference, monocular distance estimation, IMU attitude tracking, and segmented video recording run entirely on-device without telemetry leakage or internet dependency.
* **Physical Mounting Agility (Upright & Inverted)**: Native gravity auto-detection adapts the HUD, target boxes, perspective virtual lane carpet, and real road perception to windshield suction cup mounts (inverted/ceiling-mounted) and desktop/dashboard mounts (upright) seamlessly.

---

## 2. Feature Matrix

| Subsystem | Key Capabilities | Trigger & Execution Logic |
| :--- | :--- | :--- |
| **1. Forward Object Tracking** | Detects & tracks 4 primary target classes: `Car`, `Bus/Truck`, `Motorcycle/Bicycle`, `Pedestrian`. | • YOLO11 / YOLOv8 NPU-accelerated detection<br>• ByteTrack persistent `Track ID` attribution<br>• Main driving corridor ROI filtering |
| **2. FCW & Monocular Ranging** | Ground-plane projective distance estimation + Time-To-Collision (TTC) grading. | • $TTC < 1.8\text{s}$: **CRITICAL** (Red HUD banner + audio buzzer)<br>• $TTC < 2.8\text{s}$ & $D \le 25\text{m}$: **WARNING** (Yellow caution badge)<br>• Stationary following: Silent |
| **3. Virtual Lane Carpet & Road Perception** | Real asphalt boundary detection + dynamic curvature perspective lane corridor. | • Dynamically bends along with IMU yaw rate during cornering<br>• Distance tick beams at 5m, 10m, and 20m<br>• Road perception confidence badge `[LANE/ROAD xx%]` |
| **4. G-Sensor Emergency Locking** | Multi-axis high-G shock and harsh braking monitoring. | • Total acceleration $\ge 1.5\text{G}$ or emergency brake deceleration immediately locks current segment into `locked/` folder to prevent FIFO overwrite |
| **5. Loop Recording & Dark HUD** | Segmented streaming video recording + high-contrast ADAS HUD. | • 3-minute seamless cyclic video chunks<br>• FIFO auto-rotation keeps storage healthy<br>• Dark HUD showing recording status, SoC temperature, G-force, and targets |

---

## 3. Hardware Architecture & Mounting Modes

### Hardware Specifications
* **SoC**: AX630C / AX650N family with dual-core RISC-V / ARM CPU
* **NPU**: 1.0 TOPS @ INT8 (runs quantized YOLO11 models with sub-35ms inference)
* **IMU**: STMicroelectronics LSM6DSOWTR 6-axis Gyroscope & Accelerometer
* **Optics**: High Dynamic Range wide-angle lens (FOV-H $85^\circ$, FOV-V $65^\circ$)
* **Display**: Integrated 640x480 color capacitive touch display

### Windshield Mounting Flexibility (`MOUNT_MODE`)
Configurable in `config.py` via `MOUNT_MODE = "auto"` (`"auto"`, `"upright"`, `"inverted"`):
* **Upright (Dashboard mount)**: Normal camera optics with standard HUD status bars.
* **Inverted (Windshield suction cup mount)**: IMU detects inverted gravity vector ($a_y < -3.0\text{ m/s}^2$). The HUD automatically swaps top/bottom status bars, inverts ground-plane lane projections, and rotates text labels by 180° for 100% human-readable, upright display without requiring lossy CPU image flipping.

---

## 4. Development & Quick Start

### 4.1 Running Unit Tests (Desktop Simulation)
The codebase includes comprehensive unit tests with mocks for camera, NPU, and display interfaces:

```bash
# Clone the repository
git clone git@github.com:Zhgong/maixcam-dashcam-internal.git
cd maixcam-dashcam

# Install test dependencies
pip install pytest opencv-python numpy

# Run test suite
PYTHONPATH=. pytest tests/
```

### 4.2 One-Click Deployment to Device
Ensure your development PC and MaixCAM 2 are on the same Wi-Fi / USB-RNDIS subnet:

```bash
# Deploy to board (default IP or specify board IP)
bash deploy.sh 10.196.232.1
```
The deployment script executes:
1. Local test matrix gate verification (must pass 100%).
2. Remote deployment package staging into `/maixapp/apps/camp_dashcam/`.
3. Bytecode pre-compilation check on device.
4. App index refresh and live end-to-end smoke test launch.

---

## 5. Repository Documentation Index

* 📐 [Architecture & Mathematical Models (docs/ARCHITECTURE.md)](docs/ARCHITECTURE.md): Monocular ranging geometry derivation, TTC matrix, and multi-thread architecture.
* 🛠️ [Development & TDD Testing Matrix (docs/DEVELOPMENT_GUIDE.md)](docs/DEVELOPMENT_GUIDE.md): Testing methodology, mock strategies, and quality gates.
* 🛡️ [Security & Desensitization Gate (.agents/AGENTS.md)](.agents/AGENTS.md): Dual-layer pre-push and GitHub Actions security audit policies.
