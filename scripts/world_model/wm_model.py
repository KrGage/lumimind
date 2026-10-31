"""Causal Transformer 世界模型（AMRS 式 Factorized 架构，紧凑实验版）"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wm_data import D_PHYSIO

TOKEN_DIM = D_PHYSIO + 8  # physio ⊕ light(3) ⊕ feedback(2) ⊕ mask(3)


class CausalTransformerWorldModel(nn.Module):
    """
    输入: 历史 token 序列 (B, T, TOKEN_DIM) + 候选光照 (B, 3)
    输出: 预测的下一情绪 (B, 2) ∈ [0,1]²（valence, arousal）

    因果掩码保证 step_t 只能看到 step₁..t，防止未来信息泄露；
    不加位置编码（AMRS 最佳配置）。
    """

    def __init__(self, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.token_proj = nn.Linear(TOKEN_DIM, d_model)
        self.bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.light_encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(
            nn.Linear(d_model + 64, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid(),
        )

    def forward(self, history, action):
        B, T, _ = history.shape
        x = torch.cat([self.bos.expand(B, -1, -1), self.token_proj(history)], dim=1)
        L = T + 1
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        h = self.transformer(x, mask=mask)
        z_t = h[:, -1, :]
        pred = self.head(torch.cat([z_t, self.light_encoder(action)], dim=-1))
        return pred, z_t

    @torch.no_grad()
    def predict(self, history, action):
        """单样本便捷接口: history (T, TOKEN_DIM) numpy, action (3,) numpy → (v, a, conf)"""
        import numpy as np

        self.eval()
        h = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        preds = np.array([self(h, a)[0][0].numpy() for _ in range(8)])  # MC-dropout 方差作置信度
        mean = preds.mean(0)
        conf = float(1.0 - min(1.0, preds.std(0).mean() * 8))
        return float(mean[0]), float(mean[1]), conf


def train(model, samples, epochs=4, batch=64, lr=3e-4, max_len=20, device="cpu", log=print):
    """samples: [(history, action, target_va), ...]"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(samples)
    model.train()
    for ep in range(epochs):
        idx = torch.randperm(n)
        total, seen = 0.0, 0
        for i in range(0, n, batch):
            batch_idx = idx[i:i + batch]
            # 历史长度不一：左侧零填充到 max_len（因果掩码下零 token 在最前，不影响末位 z_t）
            hist = torch.zeros(len(batch_idx), max_len, TOKEN_DIM)
            act = torch.stack([torch.tensor(samples[j][1]) for j in batch_idx]).to(device)
            tgt = torch.tensor([samples[j][2] for j in batch_idx], dtype=torch.float32).to(device)
            for k, j in enumerate(batch_idx):
                h = torch.tensor(samples[j][0][-max_len:])
                hist[k, max_len - len(h):] = h
            hist = hist.to(device)

            pred, _ = model(hist, act)
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch_idx)
            seen += len(batch_idx)
        log(f"  epoch {ep + 1}/{epochs}  MSE={total / seen:.4f}")
