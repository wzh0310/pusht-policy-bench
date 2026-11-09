"""BC 训练脚本：监督学习，损失 = 归一化动作空间上的 MSE。

用法:
    python train.py --config config.yaml
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import BCDataset
from model import BCPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--max-steps", type=int, default=None, help="调试用：限制每 epoch 步数")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["train"]["seed"])

    total_episodes = json.load(
        open(Path(cfg["dataset"]["root"]) / "meta" / "info.json")
    )["total_episodes"]
    n_val = min(8, total_episodes // 10)
    val_episodes = list(range(total_episodes - n_val, total_episodes))
    train_episodes = list(range(total_episodes - n_val))

    feat_keys = {k: v for k, v in cfg["dataset"].items() if k not in ("root", "repo_id", "augment")}
    train_ds = BCDataset(cfg["dataset"]["root"], episodes=train_episodes,
                         augment=cfg["dataset"].get("augment", False), **feat_keys)
    val_ds = BCDataset(cfg["dataset"]["root"], episodes=val_episodes, **feat_keys)

    g = torch.Generator(); g.manual_seed(cfg["train"]["seed"])
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          num_workers=cfg["train"]["num_workers"], drop_last=True,
                          generator=g, persistent_workers=cfg["train"]["num_workers"] > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"], persistent_workers=False)

    sample = train_ds[0]
    model = BCPolicy(
        state_dim=sample["state"].shape[-1],
        action_dim=sample["action"].shape[-1],
        **cfg["model"],
    ).to(cfg["train"]["device"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M | 训练集 {len(train_ds)} 帧 | 验证集 {len(val_ds)} 帧")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    best_val = float("inf")
    import os
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = Path(__file__).parent / "outputs" / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for step, batch in enumerate(train_dl):
            if args.max_steps and step >= args.max_steps:
                break
            batch = {k: v.to(cfg["train"]["device"]) for k, v in batch.items()}
            loss = model.loss(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item(); n += 1
            if (step + 1) % 50 == 0:
                print(f"  epoch {epoch} step {step+1}/{len(train_dl)} loss {running/n:.4f}")

        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(cfg["train"]["device"]) for k, v in batch.items()}
                val_loss += model.loss(batch).item(); vn += 1
        val_loss /= max(vn, 1)
        print(f"epoch {epoch}: train_loss {running/max(n,1):.4f} | val_loss {val_loss:.4f} | {time.time()-t0:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "normalization": {
                    "state_mean": train_ds.state_mean, "state_std": train_ds.state_std,
                    "action_mean": train_ds.action_mean, "action_std": train_ds.action_std,
                    "img_mean": train_ds.img_mean, "img_std": train_ds.img_std,
                },
                "model_kwargs": {
                    "state_dim": sample["state"].shape[-1],
                    "action_dim": sample["action"].shape[-1],
                    **cfg["model"],
                },
            }, out_dir / "checkpoint.pt")

    print(f"训练完成，最优验证 loss: {best_val:.4f}，权重保存在 {out_dir/'checkpoint.pt'}")


if __name__ == "__main__":
    main()
