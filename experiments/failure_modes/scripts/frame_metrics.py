"""失败轨迹逐帧分析共用工具（像素运动代理指标 + 抽帧拼图）。

口径与 experiments/flow_vs_diffusion/scripts/analyze_smoothness.py 一致：
逐帧灰度帧差 -> 运动能量 d_t；二阶差分 -> jerk 代理；按动作块长度切分
比较块边界与块内抖动，用于定位「开环执行整块 -> 重规划」引入的不连续。
"""
import glob
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = (30, 41, 59)
MUTED = (100, 116, 139)
GRID = (226, 232, 240)


def _cjk_font(path, size):
    try:
        f = ImageFont.truetype(path, size)
    except OSError:
        return None

    def glyph(ch):
        im = Image.new("L", (48, 48), 0)
        ImageDraw.Draw(im).text((4, 4), ch, font=f, fill=255)
        return im.tobytes()

    if glyph("扩") != glyph("散") and glyph("流") != glyph("匹"):
        return f
    return None


def cjk_font(size):
    out = subprocess.run(["fc-list", ":lang=zh", "-f", "%{file}\n"],
                         capture_output=True, text=True).stdout.splitlines()
    cands = [x for x in dict.fromkeys(out) if x.strip() and Path(x).exists()]
    cands += glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    cands += glob.glob("/usr/share/fonts/**/*.TTF", recursive=True)
    for p in cands:
        f = _cjk_font(p, size)
        if f:
            return f
    return ImageFont.load_default()


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def compute_metrics(frames, chunk=32, thumb_count=6):
    """返回逐帧序列、汇总指标与等间隔抽帧。"""
    gray = [cv2.cvtColor(cv2.resize(f, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            for f in frames]
    motion = [float(np.mean(np.abs(gray[t] - gray[t - 1])))
              for t in range(1, len(gray))]
    jerk = [abs(motion[t] - 2 * motion[t - 1] + motion[t - 2])
            for t in range(2, len(motion))]
    med = float(np.median(jerk)) if jerk else 0.0
    spikes = sum(1 for x in jerk if med > 0 and x > 3 * med)

    # 动作块边界：开环执行 chunk 步后重规划，边界处最易出现不连续
    boundary, inner = [], []
    for i, x in enumerate(jerk):
        (boundary if (i + 2) % chunk == 0 else inner).append(x)

    per_frame = [{"t": t + 1, "motion": round(motion[t], 5)} for t in range(len(motion))]
    for i, x in enumerate(jerk):
        per_frame[i + 2]["jerk"] = round(x, 5)

    idx = [round(k * (len(frames) - 1) / max(1, thumb_count - 1))
           for k in range(thumb_count)] if frames else []
    thumbs = [frames[i] for i in idx]

    return {
        "n_frames": len(frames),
        "chunk": chunk,
        "motion_mean": round(float(np.mean(motion)), 5),
        "motion_p95": round(float(np.percentile(motion, 95)), 5),
        "motion_max": round(float(np.max(motion)), 5),
        "jerk_mean": round(float(np.mean(jerk)), 5),
        "jerk_p95": round(float(np.percentile(jerk, 95)), 5),
        "jerk_max": round(float(np.max(jerk)), 5),
        "spike_rate": round(spikes / max(1, len(jerk)), 5),
        "stall_ratio": round(sum(1 for x in motion
                                 if x < 0.5 * float(np.median(motion))) / max(1, len(motion)), 4),
        "motion_first_half": round(float(np.mean(motion[:len(motion) // 2])), 5),
        "motion_second_half": round(float(np.mean(motion[len(motion) // 2:])), 5),
        "tail_motion_mean": round(float(np.mean(motion[-30:])), 5),
        "boundary_jerk_mean": round(float(np.mean(boundary)), 5) if boundary else None,
        "inner_jerk_mean": round(float(np.mean(inner)), 5) if inner else None,
        "boundary_ratio": (round(float(np.mean(boundary)) / float(np.mean(inner)), 4)
                           if boundary and inner and np.mean(inner) > 0 else None),
        "per_frame": per_frame,
    }, thumbs


def contact_sheet(rows, out_path, thumb=170, label_w=250, cols=None):
    """rows: [{"label": str, "thumbs": [BGR...]}] -> 每行一条轨迹的横向抽帧图。"""
    cols = cols or max(len(r["thumbs"]) for r in rows)
    f_label = cjk_font(17)
    f_head = cjk_font(21)
    W = label_w + cols * thumb + 20
    H = 50 + len(rows) * (thumb + 16) + 20
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 14), "失败轨迹逐帧抽帧对比（每行一条轨迹，等间隔 6 帧）",
           font=f_head, fill=INK)
    for ri, row in enumerate(rows):
        y = 50 + ri * (thumb + 16)
        d.multiline_text((14, y + thumb // 2 - 26), row["label"],
                         font=f_label, fill=INK, spacing=8)
        d.line([(label_w - 10, y), (label_w - 10, y + thumb)], fill=GRID, width=2)
        for ci, fr in enumerate(row["thumbs"][:cols]):
            rgb = cv2.cvtColor(cv2.resize(fr, (thumb, thumb)), cv2.COLOR_BGR2RGB)
            img.paste(Image.fromarray(rgb), (label_w + ci * thumb, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=90)
    return img.size
