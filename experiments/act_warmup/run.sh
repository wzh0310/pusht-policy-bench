# ACT 训练 1000 步（lerobot 0.7.1，本地 pusht 数据集）
# 环境：py312-robot (torch 2.7.1, lerobot 0.7.1)
lerobot-train \
  --dataset.repo_id=pusht \
  --dataset.root=/data/wzh/robot/datasets/pusht \
  --policy.type=act \
  --policy.repo_id=local/act_pusht_demo \
  --steps=1000 \
  --output_dir=outputs/act_demo \
  --env_eval_freq=0 --save_freq=1000 --trackio.enable=false
