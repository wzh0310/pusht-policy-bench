# 最小 BC 基线训练（20 epoch，A6000 约 3 分钟）
# 环境：py312-robot (torch 2.7.1)
cd baselines/bc && python train.py --config config.yaml

# 在线评估（gym_pusht 闭环 rollout，20 episode，导出最佳视频）
python eval_env.py --episodes 20 --out eval_results.json --video ../assets/bc_baseline_best.mp4
