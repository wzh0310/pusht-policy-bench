"""ALOHA 双臂仿真数据集格式探查（lerobot 0.7.1）。

注意：常见的旧版示例用 ``dataset.observation_keys``，0.7.1 已移除该属性，
改用 ``dataset.features`` / ``dataset.meta.info`` 查询观测与动作规格。
所有落盘路径都写成相对工作区的路径，不暴露机器绝对路径。
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parent.parent
WZH = Path(os.environ.get("WZH_ROOT", REPO.parent))
OUT = BASE / "results"
CACHE = WZH / "datasets" / "aloha_sim_insertion_human"
REPO_ID = "lerobot/aloha_sim_insertion_human"

OUT.mkdir(parents=True, exist_ok=True)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

print(f"loading {REPO_ID} (local cache: datasets/aloha_sim_insertion_human)", flush=True)
ds = LeRobotDataset(REPO_ID, root=str(CACHE), download_videos=True)
print(f"loaded: {ds.num_episodes} episodes / {ds.num_frames} frames / {ds.fps} fps", flush=True)

sample = ds[0]


def desc(v):
    if hasattr(v, "shape"):
        return {"type": "tensor", "shape": list(v.shape),
                "dtype": str(getattr(v, "dtype", ""))}
    return {"type": type(v).__name__, "value": str(v)[:120]}


features = {}
raw_features = getattr(ds, "features", None) or {}
for k, v in raw_features.items():
    if isinstance(v, dict):
        features[k] = {kk: vv for kk, vv in v.items()
                       if kk in ("dtype", "shape", "names", "fps")}
    else:
        features[k] = {"repr": str(v)[:160]}

camera_keys = sorted(k for k in features if k.startswith("observation.images"))
action_spec = features.get("action", {})
state_spec = features.get("observation.state", {})

# episode 长度分布与任务列表：直接读落盘的 parquet 元数据
episode_lengths, tasks = [], []
try:
    import pandas as pd
    ep_dir = CACHE / "meta" / "episodes"
    for p in sorted(ep_dir.rglob("*.parquet")):
        df = pd.read_parquet(p)
        col = "length" if "length" in df.columns else None
        if col:
            episode_lengths += [int(x) for x in df[col].tolist()]
    tp = CACHE / "meta" / "tasks.parquet"
    if tp.exists():
        tdf = pd.read_parquet(tp)
        for c in tdf.columns:
            if c in ("task", "name"):
                tasks += [str(x) for x in tdf[c].tolist()[:20]]
except Exception as exc:  # 元数据缺失不影响主流程
    print("meta parquet skipped:", exc, flush=True)

info = {
    "repo_id": REPO_ID,
    "lerobot_api": "0.7.1",
    "num_episodes": int(ds.num_episodes),
    "num_frames": int(ds.num_frames),
    "fps": int(ds.fps),
    "features": features,
    "sample_keys": sorted(sample.keys()),
    "sample_shapes": {k: desc(v) for k, v in sample.items()},
    "camera_keys": camera_keys,
    "n_cameras": len(camera_keys),
    "action_dim": action_spec.get("shape"),
    "action_names": action_spec.get("names"),
    "state_dim": state_spec.get("shape"),
    "state_names": state_spec.get("names"),
    "episode_lengths": episode_lengths[:20],
    "tasks": tasks[:20] or [str(sample.get("task", ""))],
    "task_text": str(sample.get("task", "")),
    "api_notes": {
        "observation_keys": "0.7.1 已移除；用 dataset.features 或 dataset.meta.info['features']",
        "num_episodes": "仍可用",
        "sample": "dataset[0] 返回 dict，图像键为 observation.images.*",
    },
}

(OUT / "aloha_probe.json").write_text(
    json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: info[k] for k in
                  ("num_episodes", "num_frames", "fps", "camera_keys",
                   "action_dim", "state_dim", "sample_keys")},
                 ensure_ascii=False, indent=2), flush=True)
print("wrote results/aloha_probe.json", flush=True)
