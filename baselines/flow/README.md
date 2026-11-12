# Flow Policy 最小实现（流匹配动作头）

lerobot 无独立 flow 策略（原生实现在 GR00T 动作头 / π0），本目录是与 `baselines/bc`
共享观测编码骨架的最小 Flow Matching 实现，用于对照 Diffusion Policy。

- 训练：`python flow_train.py --config config_flow.yaml`（2000 步约 7 分钟，A6000）
- 评估：`python flow_eval_env.py --episodes 10`（gym_pusht 闭环，开环执行 8 步块）
- 结构差异与实测数据：../../docs/flow_notes.md
