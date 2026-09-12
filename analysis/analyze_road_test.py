"""
Comprehensive Analysis Script for MaixCAM 2 Dashcam Road Test Data
"""

import os
import glob
import re
from collections import Counter
import pandas as pd
import numpy as np

DATA_DIR = "./data/road_test_20260829"
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
TELEMETRY_DIR = os.path.join(DATA_DIR, "telemetry")


def analyze_snapshots():
    files = glob.glob(os.path.join(SNAPSHOTS_DIR, "*.jpg"))
    total_snaps = len(files)
    
    classes = Counter()
    alert_levels = Counter()
    timestamps = []
    
    # Filename format: snap_YYYYMMDD_HHMMSS_id{track_id}_{class_name}_{alert_level}.jpg
    pattern = re.compile(r"snap_(\d{8}_\d{6})_id(\d+)_([a-zA-Z]+)_([a-zA-Z_]+)\.jpg")
    
    for f in files:
        fname = os.path.basename(f)
        m = pattern.match(fname)
        if m:
            t_str, t_id, cls_name, alert = m.groups()
            classes[cls_name] += 1
            alert_levels[alert] += 1
            timestamps.append(t_str)
        else:
            classes["unknown"] += 1
            
    print("=" * 60)
    print(f"📸 1. 快照图片全量分析 (Snapshots Analysis: {total_snaps} 张)")
    print("=" * 60)
    print("【目标类别分布】:")
    for cls, count in classes.most_common():
        pct = (count / total_snaps) * 100 if total_snaps else 0
        print(f"  • {cls:10s}: {count:4d} 张 ({pct:5.1f}%)")
        
    print("\n【告警级别分布】:")
    for lvl, count in alert_levels.most_common():
        pct = (count / total_snaps) * 100 if total_snaps else 0
        print(f"  • {lvl:10s}: {count:4d} 张 ({pct:5.1f}%)")

    return {
        "total_snaps": total_snaps,
        "classes": classes,
        "alert_levels": alert_levels,
        "files": files
    }


def analyze_telemetry():
    csv_files = glob.glob(os.path.join(TELEMETRY_DIR, "*.csv"))
    csv_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    
    print("\n" + "=" * 60)
    print(f"📈 2. 遥测日志深度分析 (Telemetry Analysis: {len(csv_files)} 个文件)")
    print("=" * 60)
    
    dfs = []
    for f in csv_files:
        size_kb = os.path.getsize(f) / 1024
        if size_kb < 10:  # 忽略短测试文件
            continue
        try:
            df = pd.read_csv(f)
            if len(df) > 100:
                dfs.append((os.path.basename(f), df))
        except Exception as e:
            print(f"读取 {f} 失败: {e}")
            
    for name, df in dfs:
        duration_sec = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        fps_mean = df['fps'].mean()
        temp_min, temp_max, temp_mean = df['soc_temp'].min(), df['soc_temp'].max(), df['soc_temp'].mean()
        total_g_max = df['total_g'].max()
        dyn_g_max = df['dyn_g'].max()
        
        # 目标与测距分析
        frames_with_target = df[df['target_count'] > 0]
        target_pct = (len(frames_with_target) / len(df)) * 100
        
        # 筛选有效前车距离
        valid_dist = df[(df['lead_dist'] > 0) & (df['lead_dist'] < 80)]
        dist_min = valid_dist['lead_dist'].min() if len(valid_dist) else 0
        dist_median = valid_dist['lead_dist'].median() if len(valid_dist) else 0
        
        # 告警统计
        crit_frames = len(df[df['alert_level'] == 'CRITICAL'])
        warn_frames = len(df[df['alert_level'] == 'WARNING'])
        locked_frames = len(df[df['is_locked'] == 1])
        
        print(f"\n📂 行程文件: {name} (共 {len(df):,} 帧, 时长: {duration_sec/60:.1f} 分钟)")
        print(f"  • 推理帧率 (FPS)     : 平均 {fps_mean:.1f} FPS (稳定性: {(df['fps'] >= 28).mean()*100:.1f}% >= 28FPS)")
        print(f"  • SoC 芯片温度      : 最低 {temp_min}°C, 最高 {temp_max}°C, 平均 {temp_mean:.1f}°C")
        print(f"  • 视场内有目标占比   : {target_pct:.1f}% ({len(frames_with_target):,} 帧)")
        if len(valid_dist):
            print(f"  • 前车距离 (LeadDist): 最小 {dist_min:.1f}m, 中位数 {dist_median:.1f}m")
        print(f"  • IMU 动态冲击 (G)  : 最大 Total G = {total_g_max:.2f}G, 最大 Dyn G = {dyn_g_max:.2f}G")
        print(f"  • 告警与加锁统计     : CRITICAL = {crit_frames} 帧 ({(crit_frames/len(df))*100:.2f}%), WARNING = {warn_frames} 帧, 加锁 = {locked_frames} 帧")


if __name__ == "__main__":
    analyze_snapshots()
    analyze_telemetry()
