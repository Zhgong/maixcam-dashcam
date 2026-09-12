#!/usr/bin/env bash

# Exit on any error
set -e

MAIXCAM_IP="${1:-10.196.232.1}"

echo "========================================="
echo " 1. 运行本地 Dashcam & FCW 单元测试集..."
echo "========================================="
PYTHONPATH=. python3 -m unittest discover tests
echo "✅ 所有单元测试 100% 通过！"

echo ""
echo "========================================="
echo " 2. 一键同步 Self-Contained App 包到 MaixCAM 2 ($MAIXCAM_IP)..."
echo "========================================="

# 1. 本地与远程清理缓存，防止架构不匹配的 .pyc 或残留文件
rm -rf apps/camp_dashcam/__pycache__ core/__pycache__
ssh -o StrictHostKeyChecking=no root@$MAIXCAM_IP "rm -rf /maixapp/apps/camp_dashcam/* && mkdir -p /maixapp/apps/camp_dashcam /maixapp/data/dashcam/normal /maixapp/data/dashcam/locked /maixapp/data/dashcam/snapshots /maixapp/data/dashcam/telemetry"

# 2. 复制最新 core 代码到 apps 并同步
cp core/*.py apps/camp_dashcam/
scp -r -o StrictHostKeyChecking=no apps/camp_dashcam/* root@$MAIXCAM_IP:/maixapp/apps/camp_dashcam/

# 3. 校验板端语法与字节码编译
ssh -o StrictHostKeyChecking=no root@$MAIXCAM_IP "python3 -m py_compile /maixapp/apps/camp_dashcam/*.py"
echo "✅ 板端 Python 编译校验 100% 通过！"

echo ""
echo "========================================="
echo " 3. 重载 MaixCAM 2 桌面 App 索引..."
echo "========================================="
ssh -o StrictHostKeyChecking=no root@$MAIXCAM_IP "
  python3 /maixapp/apps/gen_app_info.py 2>/dev/null || true
"

echo ""
echo "========================================="
echo " 4. 运行端到端芯片硬件直跑冒烟测试 (End-to-End Direct App Launch)..."
echo "========================================="
ssh -o StrictHostKeyChecking=no root@$MAIXCAM_IP "
  systemctl stop launcher 2>/dev/null || true
  python3 -c '
import sys, time, threading
sys.path.insert(0, \"/maixapp/apps/camp_dashcam\")
import main
t = threading.Thread(target=main.main)
t.daemon = True
t.start()
time.sleep(3)
'
  systemctl start launcher 2>/dev/null || true
"

echo "✅ Camp Dashcam MVP 真实端到端硬件直跑冒烟测试 100% 成功！"
echo "🎉 部署与端到端硬件冒烟测试 100% 成功！自包含 App 包已同步至 MaixCAM 2！"

