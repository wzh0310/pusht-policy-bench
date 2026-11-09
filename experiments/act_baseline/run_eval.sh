# 最优 checkpoint 20 episode 评估（导出全部视频需 --eval.recording=true）
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
lerobot-eval \
  --policy.path=/data/wzh/robot/outputs/act_baseline/checkpoints/last/pretrained_model \
  --env.type=pusht --env.episode_length=300 \
  --eval.n_episodes=20 --eval.recording=true \
  --policy.use_amp=false \
  --output_dir=/data/wzh/robot/outputs/act_baseline_eval
