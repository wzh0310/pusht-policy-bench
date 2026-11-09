# ACT 完整基线训练（与 BC 基线严格对齐协议）
# 数据集: 本地 pusht（相同数据）；步数: 7680（= BC 的 20 epoch × 384 步）
# 评估间隔: 每 384 步在留出集（eval_split=0.04，约 8 episodes）上算 eval_loss
lerobot-train \
  --config_path=configs/act_pusht.yaml \
  --dataset.root=/data/wzh/robot/datasets/pusht \
  --dataset.eval_split=0.04 \
  --steps=7680 --batch_size=64 \
  --log_freq=384 --eval_steps=384 \
  --save_freq=7680 --env_eval_freq=0 \
  --policy.repo_id=local/act_pusht_baseline \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --output_dir=/data/wzh/robot/outputs/act_baseline \
  --job_name=act_pusht_baseline
