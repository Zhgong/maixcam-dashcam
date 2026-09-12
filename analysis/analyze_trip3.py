"""
Trip 3 Comprehensive Telemetry & Video Cross-Verification Script
"""

import os
import glob
import pandas as pd
import numpy as np
import cv2

DATA_DIR = "./data/road_test_trip3_20260829"
TELEMETRY_DIR = os.path.join(DATA_DIR, "telemetry")
LOCKED_VIDEO_DIR = os.path.join(DATA_DIR, "locked")
NORMAL_VIDEO_DIR = os.path.join(DATA_DIR, "normal")
OUT_DIR = os.path.join(DATA_DIR, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_merge_telemetry():
    csv_files = glob.glob(os.path.join(TELEMETRY_DIR, "*.csv"))
    valid_dfs = []
    
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f).dropna(subset=['timestamp'])
            if len(df) > 50:  # 忽略极短测试
                df['source_csv'] = os.path.basename(f)
                valid_dfs.append(df)
        except Exception:
            pass
            
    if not valid_dfs:
        raise RuntimeError("No valid telemetry CSV files found!")
        
    merged = pd.concat(valid_dfs, ignore_index=True)
    return merged, valid_dfs


def analyze_metrics(df):
    total_frames = len(df)
    fps_mean = df['fps'].mean()
    temp_min, temp_max, temp_mean = df['soc_temp'].min(), df['soc_temp'].max(), df['soc_temp'].mean()
    
    # 目标出现比例
    has_target = df[df['target_count'] > 0]
    target_pct = (len(has_target) / total_frames) * 100
    
    # 有效前车车距分布
    valid_dist = df[(df['lead_dist'] > 0) & (df['lead_dist'] < 80)]
    dist_min = valid_dist['lead_dist'].min() if len(valid_dist) else 0
    dist_q25 = valid_dist['lead_dist'].quantile(0.25) if len(valid_dist) else 0
    dist_median = valid_dist['lead_dist'].median() if len(valid_dist) else 0
    dist_q75 = valid_dist['lead_dist'].quantile(0.75) if len(valid_dist) else 0
    
    # 相对逼近速度
    closing = df[df['rel_speed'] > 0.5]
    speed_mean = closing['rel_speed'].mean() if len(closing) else 0
    speed_max = closing['rel_speed'].max() if len(closing) else 0
    speed_q95 = closing['rel_speed'].quantile(0.95) if len(closing) else 0
    
    # 告警统计
    crit_mask = (df['alert_level'] == 'CRITICAL')
    warn_mask = (df['alert_level'] == 'WARNING')
    crit_count = crit_mask.sum()
    warn_count = warn_mask.sum()
    
    # IMU 冲击统计
    total_g_max = df['total_g'].max()
    dyn_g_max = df['dyn_g'].max()
    hard_shocks = (df['dyn_g'] >= 0.5).sum()
    
    return {
        "total_frames": total_frames,
        "fps_mean": fps_mean,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "temp_mean": temp_mean,
        "target_pct": target_pct,
        "dist_min": dist_min,
        "dist_q25": dist_q25,
        "dist_median": dist_median,
        "dist_q75": dist_q75,
        "speed_mean": speed_mean,
        "speed_max": speed_max,
        "speed_q95": speed_q95,
        "crit_count": crit_count,
        "crit_pct": (crit_count / total_frames) * 100,
        "warn_count": warn_count,
        "warn_pct": (warn_count / total_frames) * 100,
        "total_g_max": total_g_max,
        "dyn_g_max": dyn_g_max,
        "hard_shocks": hard_shocks
    }


def find_top_critical_episodes(df, top_n=5):
    """
    寻找持续时间最长、连续触发 CRITICAL 的典型事件区间
    """
    df = df.copy().reset_index(drop=True)
    df['is_crit'] = (df['alert_level'] == 'CRITICAL').astype(int)
    
    # 计算连续相同的区块 id
    df['block'] = (df['is_crit'] != df['is_crit'].shift(1)).cumsum()
    crit_blocks = df[df['is_crit'] == 1].groupby('block')
    
    episodes = []
    for block_id, group in crit_blocks:
        length = len(group)
        if length >= 5:  # 连续 5 帧以上（>0.15 秒）
            t_start = group['timestamp'].iloc[0]
            t_end = group['timestamp'].iloc[-1]
            source_csv = group['source_csv'].iloc[0]
            lead_dist_mean = group['lead_dist'].mean()
            rel_speed_mean = group['rel_speed'].mean()
            ttc_min = group['ttc'].min()
            episodes.append({
                "source_csv": source_csv,
                "start_idx": group.index[0],
                "end_idx": group.index[-1],
                "length_frames": length,
                "duration_sec": t_end - t_start,
                "t_start": t_start,
                "t_end": t_end,
                "lead_dist_mean": lead_dist_mean,
                "rel_speed_mean": rel_speed_mean,
                "ttc_min": ttc_min
            })
            
    episodes.sort(key=lambda x: x['length_frames'], reverse=True)
    return episodes[:top_n]


def match_episode_to_video(episode):
    """
    根据时间戳匹配对应的视频文件和在视频中的具体秒数
    """
    # 查找所有视频文件
    all_videos = glob.glob(os.path.join(LOCKED_VIDEO_DIR, "*.avi")) + glob.glob(os.path.join(NORMAL_VIDEO_DIR, "*.avi"))
    all_videos.sort()
    
    matched_video = None
    t_start = episode['t_start']
    
    # 视频文件名格式: YYYYMMDD_HHMMSS.avi
    # 遥测时间戳: timestamp (epoch float)
    import time
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_start))
    
    # 寻找修改时间或名称最接近的视频
    best_diff = float("inf")
    for v in all_videos:
        v_base = os.path.basename(v).replace(".avi", "")
        try:
            # 简单比较文件名时间差
            v_t = time.mktime(time.strptime(v_base, "%Y%m%d_%H%M%S"))
            diff = t_start - v_t
            if 0 <= diff <= 190:  # 在 3 分钟分段内
                if diff < best_diff:
                    best_diff = diff
                    matched_video = (v, diff)
        except Exception:
            pass
            
    return matched_video


def extract_incident_frames(video_path, sec_offset, out_name):
    """从视频中提取关键帧"""
    if not os.path.exists(video_path):
        return False
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame = int(sec_offset * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame))
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, frame)
        return out_path
    return False


if __name__ == "__main__":
    print("🚀 开始深度分析 Trip 3 遥测时序与对应视频...")
    merged_df, dfs = load_and_merge_telemetry()
    metrics = analyze_metrics(merged_df)
    
    print("\n" + "=" * 60)
    print("📊 1. 行程宏观体检与动力学汇总 (Trip 3 Metrics)")
    print("=" * 60)
    total_time_min = metrics['total_frames'] / (metrics['fps_mean'] * 60)
    print(f"  • 总有效遥测帧数 : {metrics['total_frames']:,} 帧 (累计约 {total_time_min:.1f} 分钟行驶)")
    print(f"  • 推理平均帧率   : {metrics['fps_mean']:.1f} FPS (全程持续高频输出)")
    print(f"  • SoC 芯片发热   : 最低 {metrics['temp_min']}°C, 最高 {metrics['temp_max']}°C, 平均 {metrics['temp_mean']:.1f}°C")
    print(f"  • 主车道目标覆盖 : {metrics['target_pct']:.1f}% 的时间前方检测到车辆/行人")
    print(f"  • 前车物理车距   : 最小 {metrics['dist_min']:.1f}m, 25分位 {metrics['dist_q25']:.1f}m, 中位数 {metrics['dist_median']:.1f}m, 75分位 {metrics['dist_q75']:.1f}m")
    print(f"  • 相对逼近速度   : 平均 {metrics['speed_mean']:.2f} m/s ({metrics['speed_mean']*3.6:.1f} km/h), 95分位 {metrics['speed_q95']:.2f} m/s ({metrics['speed_q95']*3.6:.1f} km/h)")
    print(f"  • 预警帧数统计   : CRITICAL 紧急预警 = {metrics['crit_count']:,} 帧 ({metrics['crit_pct']:.2f}%), WARNING 注意 = {metrics['warn_count']:,} 帧 ({metrics['warn_pct']:.2f}%)")
    print(f"  • IMU 动力学冲击 : 最大 Total G = {metrics['total_g_max']:.2f}G, 最大 Dyn G = {metrics['dyn_g_max']:.2f}G, 强烈颠簸事件 = {metrics['hard_shocks']} 次")

    print("\n" + "=" * 60)
    print("🎯 2. 持续时间最长的高危告警事件对齐 (Top Critical Incidents)")
    print("=" * 60)
    top_episodes = find_top_critical_episodes(merged_df, top_n=5)
    for i, ep in enumerate(top_episodes, 1):
        matched = match_episode_to_video(ep)
        v_info = f"{os.path.basename(matched[0])} 第 {matched[1]:.1f} 秒" if matched else "未精确定位到单个分段"
        print(f"\n[高危事件 #{i}]")
        print(f"  • 持续时长   : 连续 {ep['length_frames']} 帧 ({ep['duration_sec']:.2f} 秒)")
        print(f"  • 对应前车距 : 平均 {ep['lead_dist_mean']:.2f} 米, 最小 TTC: {ep['ttc_min']:.2f} 秒")
        print(f"  • 相对速度   : {ep['rel_speed_mean']:.2f} m/s ({ep['rel_speed_mean']*3.6:.1f} km/h)")
        print(f"  • 对应原始视频: {v_info}")
        
        if matched:
            img_path = extract_incident_frames(matched[0], matched[1], f"incident_{i}.jpg")
            if img_path:
                print(f"  • 已提取关键帧: {os.path.basename(img_path)}")
