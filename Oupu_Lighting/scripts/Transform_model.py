"""
时序Transformer模型
输入: 心电原始数据 (来自 data_ECG.py)
输出: 情绪分类 + 情绪强度
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 位置编码 (Positional Encoding)
# ============================================================
class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码，为序列中的每个位置添加位置信息
    使Transformer能够识别元素的顺序关系
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================
# Transformer Encoder 层
# ============================================================
class TransformerEncoder(nn.Module):
    """
    多头自注意力层 + 前馈网络
    捕捉心电信号中的时序依赖关系
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        return self.transformer_encoder(x)


# ============================================================
# ECG信号特征提取 (1D CNN前端)
# ============================================================
class CNNFeatureExtractor(nn.Module):
    """
    1D卷积网络，从原始心电信号中提取局部特征
    保留时间分辨率的同时捕获心电波形形态特征
    """
    def __init__(self, input_channels=1, cnn_channels=[64, 128, 256], kernel_size=7):
        super().__init__()
        layers = []
        in_ch = input_channels

        for out_ch in cnn_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.2))
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)  # (batch, channels, seq_len)


# ============================================================
# 时序Transformer分类模型
# ============================================================
class EmotionTransformer(nn.Module):
    """
    主模型:
    1. CNN前端: 提取ECG局部特征
    2. 位置编码: 注入序列位置信息
    3. Transformer编码器: 捕获长程时序依赖
    4. 分类头: 情绪类别 + 强度回归
    """
    def __init__(
        self,
        input_dim=1,
        cnn_channels=[64, 128, 256],
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=4,
        dropout=0.1,
        seq_len=2560
    ):
        super().__init__()
        self.d_model = d_model

        # CNN特征提取
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=input_dim,
            cnn_channels=cnn_channels
        )

        # 将CNN输出映射到d_model维度
        self.cnn_to_model = nn.Linear(cnn_channels[-1], d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len // 8 + 10, dropout=dropout)

        # Transformer编码器
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # 情感分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # 情绪强度回归头
        self.intensity_regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # -> (batch, input_dim, seq_len)

        # CNN特征提取
        x = self.cnn_extractor(x)  # (batch, cnn_channels[-1], seq_len//8)

        x = x.permute(0, 2, 1)  # -> (batch, seq_len//8, cnn_channels[-1])

        # 映射到d_model
        x = self.cnn_to_model(x)  # (batch, seq_len//8, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer(x)  # (batch, seq_len//8, d_model)

        # 聚合: 取所有时间步的加权平均
        x = x.mean(dim=1)  # (batch, d_model)

        # 分类 & 回归
        emotion_class = self.classifier(x)      # (batch, num_classes)
        intensity = self.intensity_regressor(x)  # (batch, 1)

        return emotion_class, intensity.squeeze(-1)


# ============================================================
# 模型封装与工具函数
# ============================================================
def build_model(
    input_dim=1,
    cnn_channels=[64, 128, 256],
    d_model=256,
    nhead=8,
    num_layers=4,
    num_classes=4,
    seq_len=2560,
    device=None
):
    """
    构建模型实例
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EmotionTransformer(
        input_dim=input_dim,
        cnn_channels=cnn_channels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        seq_len=seq_len
    ).to(device)

    return model


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 训练步骤示例
# ============================================================
def train_step(model, batch_x, batch_y_class, batch_y_intensity, optimizer, criterion_cls, criterion_reg, device='cpu'):
    """
    单步训练
    """
    model.train()
    batch_x = batch_x.to(device)
    batch_y_class = batch_y_class.to(device)
    batch_y_intensity = batch_y_intensity.to(device)

    optimizer.zero_grad()

    pred_class, pred_intensity = model(batch_x)

    # 分类损失
    loss_cls = criterion_cls(pred_class, batch_y_class)

    # 回归损失
    loss_reg = criterion_reg(pred_intensity, batch_y_intensity)

    # 总损失
    loss = loss_cls + loss_reg

    loss.backward()
    optimizer.step()

    return loss.item(), loss_cls.item(), loss_reg.item()


if __name__ == "__main__":
    print("=" * 60)
    print("EmotionTransformer 模型测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(device=device)
    model.eval()
    total_params = count_parameters(model)
    print(f"\n模型参数量: {total_params:,}")
    print(f"运行设备: {device}")

    # 模拟输入: batch=4, seq_len=2560 (假设4秒ECG@640Hz)
    test_input = torch.randn(4, 2560, 1).to(device)
    with torch.no_grad():
        pred_class, pred_intensity = model(test_input)

    print(f"输入shape: {test_input.shape}")
    print(f"情绪分类输出shape: {pred_class.shape}  (batch, 4分类)")
    print(f"情绪强度输出shape: {pred_intensity.shape} (batch,)")
    print(f"\n分类 logits:\n{pred_class[0]}")
    print(f"\n强度预测:\n{pred_intensity[0]:.4f}")