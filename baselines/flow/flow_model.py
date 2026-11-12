"""最小化 Flow Policy（流匹配动作头）。

与 Diffusion Policy 的对照（详见 docs/flow_notes.md）：
- Diffusion：加噪/去噪的随机微分方程，预测"当前样本里的噪声 ε"，推理 N 步随机采样；
- Flow matching：噪声→数据走一条**确定性直线插值** ODE，预测"速度场 v=x1-x0"，
  推理 N 步 Euler 积分——没有噪声调度表（beta schedule），步数可以压得更低。

网络：与 BC v2 完全相同的观测编码（ResNet18 + 状态 MLP + 融合 trunk），
动作头换成"速度场预测网络"：1D 时序卷积 + 时间步 FiLM 条件注入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_resnet18_encoder(out_dim: int) -> nn.Module:
    from torchvision.models import resnet18, ResNet18_Weights

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Sequential(nn.Linear(net.fc.in_features, out_dim), nn.ReLU(inplace=True))
    return net


class TimeEmbedding(nn.Module):
    """扩散步数/流时间的位置编码 + MLP，与 Diffusion 的 SinusoidalPosEmb 同构。"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B,) ∈ [0,1]
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))


class VelocityField(nn.Module):
    """v(x_t, t, cond)：对动作序列逐位置做 1D 卷积，时间嵌入 FiLM 调制。"""

    def __init__(self, action_dim: int, cond_dim: int, hidden: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(action_dim, hidden)
        self.time_emb = TimeEmbedding(64)
        self.film = nn.Linear(64, hidden * 2)
        self.conv1 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.out_proj = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, K, A)，t: (B,)，cond: (B, cond_dim)
        h = self.in_proj(x).transpose(1, 2)  # (B, hidden, K)
        scale, shift = self.film(self.time_emb(t)).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        c = self.cond_proj(cond)[:, :, None]
        h = F.silu(self.conv1(h) + c)
        h = h + self.conv2(h)
        return self.out_proj(h.transpose(1, 2))


class FlowPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 8,
        hidden_dim: int = 128,
        infer_steps: int = 10,
    ):
        super().__init__()
        self.horizon = horizon
        self.infer_steps = infer_steps
        self.vision_encoder = make_resnet18_encoder(hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.velocity = VelocityField(action_dim, hidden_dim, hidden_dim)

    def encode(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([self.vision_encoder(image), self.state_encoder(state)], dim=-1))

    def loss(self, batch: dict) -> torch.Tensor:
        """流匹配训练目标：MSE(v_pred, x1 - x0)。t~U[0,1]，x_t = (1-t)·x0 + t·x1。"""
        x1 = batch["action"]  # (B, K, A) 干净动作块
        x0 = torch.randn_like(x1)  # 先验噪声
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        cond = self.encode(batch["image"], batch["state"])
        v_pred = self.velocity(xt, t, cond)
        return F.mse_loss(v_pred, x1 - x0)

    @torch.no_grad()
    def sample(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Euler 积分：x_{t+Δ} = x_t + Δ·v(x_t, t)，从纯噪声走到动作块。"""
        cond = self.encode(image, state)
        x = torch.randn(image.shape[0], self.horizon, self.velocity.out_proj.out_features,
                        device=image.device)
        for i in range(self.infer_steps):
            t = torch.full((x.shape[0],), i / self.infer_steps, device=x.device)
            x = x + (1.0 / self.infer_steps) * self.velocity(x, t, cond)
        return x  # (B, K, A)
