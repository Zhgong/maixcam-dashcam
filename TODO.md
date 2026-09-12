# 📋 MaixCAM 2 Dashcam TODO & Evolution Roadmap

## 📌 Pending Tasks

### 1. English Documentation
- [ ] **English README (`README_EN.md`)**
  - Provide a full English translation of the project overview, hardware specs (1.0 TOPS NPU, VPU, LSM6DSOWTR IMU).
  - Document setup, installation, deployment, and testing workflows (`deploy.sh`, `pytest`).
  - Add quick start guides for desktop emulation vs. real on-device hardware execution.

---

### 2. Architecture & Data Flow Modular Refactoring
- [ ] **Current State & Data Flow Review**
  - **Issue**: `apps/camp_dashcam/main.py` contains a monolithic main loop handling camera acquisition, NPU inference, road perception, FCW calculation, telemetry logging, video recording, HUD rendering, screen display, and touch handling in a single synchronous cycle.
  - **Duplication**: `core/` and `apps/camp_dashcam/` have parallel file copies (`fcw_tracker.py`, `hud_renderer.py`, `road_detector.py`, etc.) that currently need manual synchronization.
- [ ] **Decoupled Multi-threaded / Pipeline Design**
  - **Perception Pipeline**: Dedicated thread for camera acquisition + NPU YOLO detection + Road edge extraction.
  - **Kinematics & Tracking Pipeline**: IMU gravity orientation + ByteTrack + FCW/TTC calculation.
  - **Display & HUD Pipeline**: Independent rendering loop ensuring 30+ FPS UI refresh without being blocked by flash I/O or model inference hiccups.
  - **Storage & Telemetry Pipeline**: Async queue-based MJPG chunking and CSV telemetry logging to prevent storage latency from stuttering the UI/detection loop.
- [ ] **Packaging & Single Source of Truth**
  - Make `core/` the single canonical package, with `deploy.sh` cleanly packaging and generating the app bundle directly into `apps/camp_dashcam/`.
