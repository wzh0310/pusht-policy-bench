"""生成面试展示素材：步数消融双面板对比图 + Flow 1 步轨迹 GIF。

数据全部读自 results/ablation_table.csv 与 assets/flow_1step_best.mp4，可复现。
"""
import csv
import glob
import math
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parent.parent
ASSETS = REPO / "assets"
R = BASE / "results"

BLUE = (37, 99, 235)
RED = (220, 38, 38)
GRID = (226, 232, 240)
INK = (30, 41, 59)
MUTED = (100, 116, 139)


def _cjk_font(path, size):
    """只接受真正含中文字形的字体（fc-match 可能返回无 CJK 字形的 fallback）。"""
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


def font_path(size):
    out = subprocess.run(["fc-list", ":lang=zh", "-f", "%{file}\n"],
                         capture_output=True, text=True).stdout.splitlines()
    candidates = [x for x in dict.fromkeys(out) if x.strip() and Path(x).exists()]
    candidates += glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/*.TTF", recursive=True)
    for path in candidates:
        f = _cjk_font(path, size)
        if f:
            return f
    return ImageFont.load_default()


F_TITLE = font_path(30)
F_LABEL = font_path(19)
F_TICK = font_path(15)
F_NOTE = font_path(15)

with (R / "ablation_table.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
flow = [(int(r["infer_steps"]), float(r["replan_ms"]), float(r["peak_coverage"]))
        for r in rows if r["algo"] == "Flow Matching"]
diff = [(int(r["infer_steps"]), float(r["replan_ms"]), float(r["peak_coverage"]))
        for r in rows if r["algo"] == "Diffusion Policy"]

W, H = 1300, 610
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
d.text((60, 28), "扩散 vs 流匹配：推理步数消融", font=F_TITLE, fill=INK)
d.text((60, 70), "PushT-v0 ｜ 50 episodes × 300 步 ｜ 同种子 ｜ RTX A6000", font=F_NOTE, fill=MUTED)

Y0, Y1 = 165, 455
PANELS = [(60, 610), (690, 1240)]


def frame(x0, x1, title, ylabel):
    d.rounded_rectangle([x0, 110, x1, 510], radius=10, outline=GRID, width=2)
    d.text((x0 + 16, 116), title, font=F_LABEL, fill=INK)
    d.text((x0 + 16, 490), ylabel, font=F_NOTE, fill=MUTED)


def lx(s, x0, x1):
    return x0 + 55 + (s - 1) / 49.0 * (x1 - x0 - 85)


def draw_legend(x0, y0):
    for i, (color, name) in enumerate(((BLUE, "Flow Matching"), (RED, "Diffusion Policy"))):
        yy = y0 + i * 24
        d.line([(x0, yy), (x0 + 26, yy)], fill=color, width=4)
        d.ellipse([x0 + 10, yy - 4, x0 + 18, yy + 4], fill=color)
        d.text((x0 + 34, yy - 11), name, font=F_TICK, fill=INK)


# 左：延迟（对数轴 1–1000 ms）
x0, x1 = PANELS[0]
frame(x0, x1, "① 推理步数 → 每次重规划延迟", "步数（Euler / 去噪）")
for v in (1, 10, 100, 1000):
    y = Y1 - math.log10(v) / 3 * (Y1 - Y0)
    d.line([(x0 + 55, y), (x1 - 20, y)], fill=GRID, width=1)
    d.text((x0 + 8, y - 9), f"{v:g}", font=F_TICK, fill=MUTED)
for s in (1, 5, 10, 20, 50):
    x = lx(s, x0, x1)
    d.text((x - 8, Y1 + 8), str(s), font=F_TICK, fill=MUTED)
for data, color in ((flow, BLUE), (diff, RED)):
    pts = [(lx(s, x0, x1), Y1 - math.log10(ms) / 3 * (Y1 - Y0)) for s, ms, _ in data]
    d.line(pts, fill=color, width=3, joint="curve")
    for i, ((x, y), (s, ms, _)) in enumerate(zip(pts, data)):
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
        if color == RED or i in (0, len(pts) - 1):   # Flow 点过密，只标首尾
            d.text((x + 7, y - 20), f"{ms:.1f}", font=F_TICK, fill=color)
draw_legend(x0 + 60, 178)

# 右：峰值覆盖率
x0, x1 = PANELS[1]
frame(x0, x1, "② 推理步数 → 峰值覆盖率", "步数（Euler / 去噪）")
for v in (0.4, 0.5, 0.6, 0.7, 0.8):
    y = Y1 - (v - 0.4) / 0.4 * (Y1 - Y0)
    d.line([(x0 + 55, y), (x1 - 20, y)], fill=GRID, width=1)
    d.text((x0 + 8, y - 9), f"{v:.1f}", font=F_TICK, fill=MUTED)
for s in (1, 5, 10, 20, 50):
    x = lx(s, x0, x1)
    d.text((x - 8, Y1 + 8), str(s), font=F_TICK, fill=MUTED)
for data, color in ((flow, BLUE), (diff, RED)):
    pts = [(lx(s, x0, x1), Y1 - (c - 0.4) / 0.4 * (Y1 - Y0)) for s, _, c in data]
    d.line(pts, fill=color, width=3, joint="curve")
    for i, ((x, y), (s, _, c)) in enumerate(zip(pts, data)):
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
        if color == RED or i in (0, len(pts) - 1):
            d.text((x + 7, y - 20), f"{c:.3f}", font=F_TICK, fill=color)
draw_legend(x0 + 300, 178)

d.text((60, 545), "结论：单步成本差 16.2×；Flow 少步数即最优，Diffusion 步数换精度但延迟线性增长。50ep 成功率 CI 重叠，不用于排序。",
       font=F_NOTE, fill=INK)
d.text((60, 572), "数据源：experiments/flow_vs_diffusion/results/ablation_table.csv ｜ 生成：scripts/make_figures.py",
       font=F_NOTE, fill=MUTED)

png = ASSETS / "flow_vs_diffusion_ablation.png"
img.save(png, optimize=True)
print("wrote", png, img.size)

# GIF：从 mp4 每 5 帧抽一帧
cap = cv2.VideoCapture(str(ASSETS / "flow_1step_best.mp4"))
rgb_frames, i = [], 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i % 5 == 0:
        rgb = cv2.cvtColor(cv2.resize(f, (256, 256)), cv2.COLOR_BGR2RGB)
        rgb_frames.append(Image.fromarray(rgb))
    i += 1
cap.release()
if rgb_frames:
    pal = rgb_frames[0].quantize(colors=128)
    gif_frames = [im.quantize(palette=pal) for im in rgb_frames]
    gif = ASSETS / "flow_1step_best.gif"
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:],
                       duration=100, loop=0, optimize=True)
    print("wrote", gif, len(gif_frames), "frames")
