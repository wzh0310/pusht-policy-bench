#!/usr/bin/env python
"""从 ACT/MetaWorld 训练日志提取 eval_loss 曲线 -> 双面板图 + 脱敏指标 JSON。

用法：在仓库根执行 python experiments/act_metaworld/scripts/plot_train_curves.py
输入：experiments/act_metaworld/train_log_{v1,v2}.txt（仓库内，已脱敏）
输出：assets/act_metaworld_train_curves.png
      experiments/act_metaworld/results/train_curves.json
"""
import glob
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent
REPO = BASE.parent.parent
ASSETS = REPO / "assets"
OUT = BASE / "results"

LOGS = {
    "v1 (3000 steps)": BASE / "train_log_v1.txt",
    "v2 (20000 steps)": BASE / "train_log_v2.txt",
}
STEP_RE = re.compile(r"step (\d+): eval_loss=([\d.]+)")
TS_RE = re.compile(r"INFO (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def cjk_font(size):
    """挑一个真正含中文字形的字体，避免渲染成豆腐块。"""
    out = subprocess.run(["fc-list", ":lang=zh", "-f", "%{file}\n"],
                         capture_output=True, text=True).stdout.splitlines()
    cands = [x for x in dict.fromkeys(out) if x.strip() and Path(x).exists()]
    cands += glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    cands += glob.glob("/usr/share/fonts/**/*.TTF", recursive=True)
    for p in cands:
        try:
            f = ImageFont.truetype(p, size)
        except OSError:
            continue
        im = Image.new("L", (48, 48), 0)
        ImageDraw.Draw(im).text((4, 4), "训练", font=f, fill=255)
        if im.tobytes() != Image.new("L", (48, 48), 0).tobytes():
            return f
    return ImageFont.load_default()


def parse(path):
    text = path.read_text(errors="ignore").replace("\r", "\n")
    pts = [(int(s), float(v)) for s, v in STEP_RE.findall(text)]
    seen, uniq = set(), []
    for s, v in pts:
        if s not in seen:
            seen.add(s)
            uniq.append((s, v))
    uniq.sort()
    times = sorted(set(TS_RE.findall(text)))
    return uniq, times


def main():
    data = {}
    for name, path in LOGS.items():
        pts, times = parse(path)
        assert pts, f"未解析到 eval_loss 点: {path}"
        data[name] = pts
        print(name, "points:", len(pts), "first:", pts[0], "last:", pts[-1],
              "min:", min(pts, key=lambda x: x[1]))
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {
        "experiment": "act_metaworld_train_curves",
        "env": "MetaWorld push-v3 (MuJoCo)",
        "metric": "eval_loss (每 300 步 v1 / 每 1000 步 v2)",
        "source_logs": ["experiments/act_metaworld/train_log_v1.txt",
                        "experiments/act_metaworld/train_log_v2.txt"],
        "figure": "assets/act_metaworld_train_curves.png",
        "versions": {},
    }
    for name, pts in data.items():
        key = name.split(" ")[0]
        metrics["versions"][key] = {
            "label": name,
            "n_points": len(pts),
            "last_step": pts[-1][0],
            "loss_first": pts[0][1],
            "loss_last": pts[-1][1],
            "loss_min": min(v for _, v in pts),
            "loss_min_step": min(pts, key=lambda x: x[1])[0],
            "reduction_pct": round(100 * (pts[0][1] - pts[-1][1]) / pts[0][1], 2),
            "points": [{"step": s, "eval_loss": v} for s, v in pts],
        }
    (OUT / "train_curves.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    W, H = 1000, 660
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ft = cjk_font(23)
    fs = cjk_font(17)
    fx = cjk_font(14)

    d.text((60, 22), "ACT 在 MetaWorld push-v3（MuJoCo）的训练曲线", fill=(30, 41, 59), font=ft)
    d.text((60, 56), "eval_loss 随训练步数下降；v2 加长训练后离线指标更低，但在线成功率未提升",
           fill=(100, 116, 139), font=fx)

    panels = [(60, 100, "v1 (3000 steps)", (37, 99, 235)),
              (60, 375, "v2 (20000 steps)", (220, 38, 38))]
    for px, py, name, color in panels:
        pts = data[name]
        pw, ph = 880, 220
        x0, y0 = px + 60, py + 30
        xs = [s for s, _ in pts]
        ys = [v for _, v in pts]
        xmax = max(xs)
        ymin, ymax = min(ys), max(ys)
        span = (ymax - ymin) or 1
        ylo, yhi = ymin - 0.15 * span, ymax + 0.15 * span

        def X(s): return x0 + s / xmax * (pw - 80)
        def Y(v): return y0 + ph - 45 - (v - ylo) / (yhi - ylo) * (ph - 70)

        d.rectangle([x0, y0, x0 + pw - 80, y0 + ph - 45], outline=(226, 232, 240), width=1)
        for frac in (0.0, 0.5, 1.0):
            v = ylo + frac * (yhi - ylo)
            yy = Y(v)
            d.line([(x0, yy), (x0 + pw - 80, yy)], fill=(241, 245, 249))
            d.text((x0 - 56, yy - 9), f"{v:.3f}", fill=(100, 116, 139), font=fx)
        poly = [(X(s), Y(v)) for s, v in pts]
        d.line(poly, fill=color, width=2, joint="curve")
        for x, y in poly:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        d.text((px, py), f"{name}   {len(pts)} 个记录点", fill=(30, 41, 59), font=fs)
        d.text((x0, y0 + ph - 40), f"step 0", fill=(100, 116, 139), font=fx)
        d.text((x0 + pw - 155, y0 + ph - 40), f"step {xmax}", fill=(100, 116, 139), font=fx)
        first, last = pts[0][1], pts[-1][1]
        d.text((x0 + 12, y0 + 10), f"{first:.4f} → {last:.4f}", fill=color, font=fs)

    ASSETS.mkdir(parents=True, exist_ok=True)
    img.save(ASSETS / "act_metaworld_train_curves.png")
    print("figure ->", ASSETS / "act_metaworld_train_curves.png")
    print("metrics ->", OUT / "train_curves.json")


if __name__ == "__main__":
    main()
