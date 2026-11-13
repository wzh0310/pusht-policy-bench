#!/usr/bin/env python
"""解析 ACT/ALOHA 训练日志 -> 指标 JSON + 摘要 + 损失曲线图（可复现，脱敏）。

用法：WZH_ROOT=<workspace-root> python scripts/parse_train_log.py
默认读取 $WZH_ROOT/outputs/act_aloha_demo_train.log，
结果写入 experiments/act_aloha/results/ 与仓库 assets/。
"""
import glob
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parent.parent
WZH = Path(os.environ.get("WZH_ROOT", REPO.parent))
LOG = Path(os.environ.get("TRAIN_LOG", WZH / "outputs" / "act_aloha_demo_train.log"))
OUT = BASE / "results"
ASSETS = REPO / "assets"

STEP_RE = re.compile(
    r"step:(?P<step>\d+) smpl:(?P<smpl>[\d.KM]+) ep:(?P<ep>\d+) epch:(?P<epoch>[\d.]+) "
    r"loss:(?P<loss>[\d.]+) grdn:(?P<grad_norm>[\d.]+) lr:(?P<lr>[\d.eE+-]+) "
    r"data_s:(?P<data_s>[\d.]+) prep_s:(?P<prep_s>[\d.]+) updt_s:(?P<updt_s>[\d.]+) "
    r"step_s:(?P<step_s>[\d.]+) smp/s:(?P<smp_s>[\d.]+) mem_gb:(?P<mem_gb>[\d.]+) "
    r"l1_loss:(?P<l1_loss>[\d.]+) kld_loss:(?P<kld_loss>[\d.]+)"
)
TS_RE = re.compile(r"INFO (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _to_num(k, v):
    v = v.strip()
    if k == "smpl":
        if v.endswith("K"):
            return int(float(v[:-1]) * 1e3)
        if v.endswith("M"):
            return int(float(v[:-1]) * 1e6)
        return int(v)
    if k in ("step", "ep"):
        return int(v)
    return float(v)


def cjk_font(size):
    out = subprocess.run(["fc-list", ":lang=zh", "-f", "%{file}\n"],
                         capture_output=True, text=True).stdout.splitlines()
    cands = [x for x in dict.fromkeys(out) if x.strip() and Path(x).exists()]
    cands += glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    cands += glob.glob("/usr/share/fonts/**/*.TTF", recursive=True)

    def ok(path):
        try:
            f = ImageFont.truetype(path, size)
        except OSError:
            return None
        im = Image.new("L", (48, 48), 0)
        ImageDraw.Draw(im).text((4, 4), "训练", font=f, fill=255)
        if im.tobytes() != Image.new("L", (48, 48), 0).tobytes():
            return f
        return None

    for p in cands:
        f = ok(p)
        if f:
            return f
    return ImageFont.load_default()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(WZH))
    except ValueError:
        return str(p)


def parse():
    text = LOG.read_text(errors="ignore")
    points, times = [], []
    for m in STEP_RE.finditer(text):
        d = {k: _to_num(k, v) for k, v in m.groupdict().items()}
        points.append(d)
    for tm in TS_RE.finditer(text):
        times.append(datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S"))
    seen, uniq = set(), []
    for d in points:
        if d["step"] not in seen:
            seen.add(d["step"])
            uniq.append(d)
    uniq.sort(key=lambda x: x["step"])
    seen_t = sorted(set(times))
    wall = (seen_t[-1] - seen_t[0]).total_seconds() if len(seen_t) >= 2 else None
    return uniq, wall


def main():
    points, wall = parse()
    assert points, f"未从日志解析到任何 step 指标: {LOG}"
    first, last = points[0], points[-1]
    metrics = {
        "experiment": "act_aloha_transfer",
        "policy": "act",
        "dataset": "lerobot/aloha_sim_insertion_human",
        "env_task": "AlohaInsertion-v0",
        "lerobot_version": "0.7.1",
        "steps_trained": last["step"],
        "batch_size": 8,
        "num_learnable_params": 51613582,
        "checkpoints": [
            "outputs/act_aloha_demo/checkpoints/001000/pretrained_model",
            "outputs/act_aloha_demo/checkpoints/002000/pretrained_model",
        ],
        "train_wall_clock_s": round(wall, 1) if wall else None,
        "loss_first": round(first["loss"], 4),
        "loss_last": round(last["loss"], 4),
        "loss_reduction_pct": round(100 * (first["loss"] - last["loss"]) / first["loss"], 2),
        "l1_first": round(first["l1_loss"], 4),
        "l1_last": round(last["l1_loss"], 4),
        "kld_first": round(first["kld_loss"], 4),
        "kld_last": round(last["kld_loss"], 4),
        "peak_mem_gb": round(max(p["mem_gb"] for p in points), 2),
        "mean_step_s": round(sum(p["step_s"] for p in points) / len(points), 4),
        "log_points": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}
            for p in points
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    lines = ["# ACT/ALOHA 训练日志摘录（每 100 步一条）", ""]
    lines.append("step,smpl,epoch,loss,l1_loss,kld_loss,grad_norm,step_s,mem_gb")
    for p in points:
        lines.append(f"{p['step']},{p['smpl']},{p['epoch']:.2f},{p['loss']:.3f},"
                     f"{p['l1_loss']:.3f},{p['kld_loss']:.3f},{p['grad_norm']:.1f},"
                     f"{p['step_s']:.3f},{p['mem_gb']:.2f}")
    (OUT / "train_log_excerpt.txt").write_text("\n".join(lines) + "\n")

    draw_curve(points)
    print("metrics ->", rel(OUT / "train_metrics.json"))
    print("excerpt ->", rel(OUT / "train_log_excerpt.txt"))
    print("figure  ->", rel(ASSETS / "act_aloha_loss.png"))
    print(f"points={len(points)} loss {first['loss']:.2f} -> {last['loss']:.2f} "
          f"(-{metrics['loss_reduction_pct']}%), wall={metrics['train_wall_clock_s']}s")


def draw_curve(points):
    W, H = 980, 420
    pad = 70
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ft = cjk_font(22)
    fl = cjk_font(16)
    ftick = cjk_font(13)

    xs = [p["step"] for p in points]
    xmin, xmax = min(xs), max(xs)

    def series(key, color, dashed=False):
        ys = [p[key] for p in points]
        ymin, ymax = min(ys), max(ys)
        ymin = ymin - 0.1 * (ymax - ymin or 1)
        ymax = ymax + 0.1 * (ymax - ymin or 1)
        def X(x): return pad + (x - xmin) / (xmax - xmin or 1) * (W - 2 * pad)
        def Y(y): return H - pad - (y - ymin) / (ymax - ymin or 1) * (H - 2 * pad)
        d.line([(X(xmin), Y(ymin)), (X(xmax), Y(ymin)), (X(xmax), Y(ymax)),
                (X(xmin), Y(ymax)), (X(xmin), Y(ymin))], fill=(226, 232, 240))
        pts = [(X(x), Y(y)) for x, y in zip(xs, ys)]
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            d.line([a, b], fill=color, width=2, joint="curve")
            if dashed:
                d.ellipse([b[0] - 3, b[1] - 3, b[0] + 3, b[1] + 3], fill=color)
    series("loss", (37, 99, 235))
    series("l1_loss", (220, 38, 38))
    series("kld_loss", (22, 163, 74), dashed=True)
    d.text((pad, 28), "ACT 在 ALOHA 双臂插入任务的训练损失（2000 步）", fill=(30, 41, 59), font=ft)
    d.text((W - 300, H - 40), "蓝=总损失  红=L1  绿=KL(虚线)", fill=(100, 116, 139), font=fl)
    d.text((pad, H - 40), f"step 0 -> {xmax}", fill=(100, 116, 139), font=ftick)
    ASSETS.mkdir(parents=True, exist_ok=True)
    img.save(ASSETS / "act_aloha_loss.png")


if __name__ == "__main__":
    main()
