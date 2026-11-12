"""Flow Policy 训练脚本（2000 步最小 demo，验证流程通畅）。"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from flow_dataset import FlowDataset
from flow_model import FlowPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config_flow.yaml"))
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["train"]["seed"])

    root = cfg["dataset"]["root"]
    info = json.load(open(Path(root) / "meta" / "info.json"))
    total = info["total_episodes"]
    n_val = min(8, total // 10)
    train_ep = list(range(total - n_val))
    val_ep = list(range(total - n_val, total))

    train_ds = FlowDataset(root, episodes=train_ep, augment=cfg["dataset"].get("augment", False),
                           **{k: v for k, v in cfg["dataset"].items()
                              if k not in ("root", "repo_id", "augment", "fps")})
    val_ds = FlowDataset(root, episodes=val_ep,
                         **{k: v for k, v in cfg["dataset"].items()
                            if k not in ("root", "repo_id", "augment", "fps")})
    g = torch.Generator(); g.manual_seed(cfg["train"]["seed"])
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          num_workers=cfg["train"]["num_workers"], drop_last=True,
                          generator=g, persistent_workers=cfg["train"]["num_workers"] > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"])

    sample = train_ds[0]
    model = FlowPolicy(state_dim=sample["state"].shape[-1], action_dim=sample["action"].shape[-1],
                       horizon=sample["action"].shape[0], **cfg["model"]).to(cfg["train"]["device"])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"训练集 {len(train_ds)} 帧 | 动作块 K={sample['action'].shape[0]}")
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    best_val, out_dir = float("inf"), Path(__file__).parent / "outputs" / "flow"
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for batch in train_dl:
            if step >= cfg["train"]["steps"]:
                break
            if args.max_steps and n >= args.max_steps:
                break
            batch = {k: v.to(cfg["train"]["device"]) for k, v in batch.items()}
            loss = model.loss(batch)
            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item(); n += 1; step += 1
            if step % 100 == 0:
                print(f"  step {step} loss {running/n:.4f}")
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(cfg["train"]["device"]) for k, v in batch.items()}
                vl += model.loss(batch).item(); vn += 1
        vl /= max(vn, 1)
        print(f"epoch {epoch}: train_loss {running/max(n,1):.4f} | val_loss {vl:.4f} | {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "model_kwargs": {"state_dim": sample["state"].shape[-1],
                                         "action_dim": sample["action"].shape[-1],
                                         "horizon": sample["action"].shape[0], **cfg["model"]},
                        "normalization": {"state_mean": train_ds.state_mean, "state_std": train_ds.state_std,
                                          "action_mean": train_ds.action_mean, "action_std": train_ds.action_std,
                                          "img_mean": train_ds.img_mean, "img_std": train_ds.img_std}},
                       out_dir / "checkpoint.pt")
        if step >= cfg["train"]["steps"]:
            break
    print(f"完成，最优验证 loss: {best_val:.4f}，权重: {out_dir/'checkpoint.pt'}")


if __name__ == "__main__":
    main()
