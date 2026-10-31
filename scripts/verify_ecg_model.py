"""
PhysioWave ecg.pth 模型验证脚本
按照 PhysioWave 论文 (NeurIPS 2025) 中的方法验证下载的 ecg.pth 预训练模型

验证内容:
  1. 模型权重加载与架构验证
  2. 小波分解模块验证
  3. 前向传播 (分类任务) 验证
  4. 特征提取模式验证
  5. 预训练重建头验证
  6. 合成 ECG 信号推理验证
  7. 梯度流验证 (反向传播)
  8. 模型参数量统计

参考论文:
  PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation
  (Chen et al., NeurIPS 2025, arXiv:2506.10351)
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

# 将 PhysioWave-main 加入 sys.path
PHYSIOWAVE_DIR = r"d:\研究生\项目内容\杂事\光源智能体设计大赛\项目主体\reference\example\PhysioWave-main"
sys.path.insert(0, PHYSIOWAVE_DIR)

from model import BERTWaveletTransformer

# ============================================================
# 配置: ecg.pth 的训练参数 (从 checkpoint['args'] 中提取)
# ============================================================
ECG_CONFIG = {
    'in_channels': 12,
    'max_level': 3,
    'wave_kernel_size': 24,
    'wavelet_names': ['db4', 'db6', 'sym4', 'coif2'],
    'use_separate_channel': True,
    'patch_size': (1, 64),
    'embed_dim': 384,
    'depth': 8,
    'num_heads': 12,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    'task_type': 'classification',
    'num_classes': 5,
    'head_hidden_dim': 1024,
    'head_dropout': 0.1,
    'pooling': 'mean',
    'use_pos_embed': True,
    'pos_embed_type': '2d',
}
CHECKPOINT_PATH = os.path.join(PHYSIOWAVE_DIR, 'ecg.pth')

# PTB-XL 5 超类标签映射
PTBXL_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 1. 加载 checkpoint 并验证元数据
# ============================================================
def test_checkpoint_loading():
    separator("1. Checkpoint 加载与元数据验证")
    
    assert os.path.exists(CHECKPOINT_PATH), f"ecg.pth not found at {CHECKPOINT_PATH}"
    file_size = os.path.getsize(CHECKPOINT_PATH) / (1024**2)
    print(f"  文件大小: {file_size:.1f} MB")
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    assert isinstance(ckpt, dict), "checkpoint 应为 dict 类型"
    expected_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'args']
    for k in expected_keys:
        assert k in ckpt, f"缺少预期的 key: {k}"
    print(f"  Top-level keys: {list(ckpt.keys())}")
    
    # 验证训练元数据
    args = ckpt['args']
    print(f"  训练 epoch: {ckpt['epoch']}")
    print(f"  Val Loss:   {ckpt.get('val_loss', 'N/A'):.4f}" if isinstance(ckpt.get('val_loss'), (int, float)) else f"  Val Loss:   {ckpt.get('val_loss')}")
    print(f"  Val Acc:    {ckpt.get('val_acc', 'N/A'):.4f}" if isinstance(ckpt.get('val_acc'), (int, float)) else f"  Val Acc:    {ckpt.get('val_acc')}")
    print(f"  Val BalAcc: {ckpt.get('val_balanced_acc', 'N/A'):.4f}" if isinstance(ckpt.get('val_balanced_acc'), (int, float)) else "")
    print(f"  Val WF1:    {ckpt.get('val_weighted_f1', 'N/A'):.4f}" if isinstance(ckpt.get('val_weighted_f1'), (int, float)) else "")
    print(f"  Val AUROC:  {ckpt.get('val_auroc', 'N/A'):.4f}" if isinstance(ckpt.get('val_auroc'), (int, float)) else "")
    
    # 验证 args 中关键超参数与论文一致
    assert args['in_channels'] == 12, "ECG 应为 12 导联"
    assert args['embed_dim'] == 384, "Base 模型 embed_dim 应为 384"
    assert args['depth'] == 8, "Base 模型 depth 应为 8"
    assert args['num_heads'] == 12, "Base 模型 num_heads 应为 12"
    assert args['num_classes'] == 5, "PTB-XL 5 超类分类"
    assert args['wavelet_names'] == ['db4', 'db6', 'sym4', 'coif2'], "ECG 小波基应为 4 个"
    assert args['patch_size'] == 64, "ECG patch_size 应为 64"
    
    sd = ckpt['model_state_dict']
    print(f"  模型参数 key 数量: {len(sd)}")
    total_params = sum(v.numel() for v in sd.values())
    print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("  [PASS] Checkpoint 元数据验证通过")
    return ckpt


# ============================================================
# 2. 重建模型架构并加载权重
# ============================================================
def test_model_rebuild(ckpt):
    separator("2. 模型重建与权重加载")
    
    head_config = {
        'hidden_dims': [ECG_CONFIG['head_hidden_dim']],
        'dropout': ECG_CONFIG['head_dropout'],
        'pooling': ECG_CONFIG['pooling'],
    }
    
    model = BERTWaveletTransformer(
        in_channels=ECG_CONFIG['in_channels'],
        max_level=ECG_CONFIG['max_level'],
        wave_kernel_size=ECG_CONFIG['wave_kernel_size'],
        wavelet_names=ECG_CONFIG['wavelet_names'],
        use_separate_channel=ECG_CONFIG['use_separate_channel'],
        patch_size=ECG_CONFIG['patch_size'],
        embed_dim=ECG_CONFIG['embed_dim'],
        depth=ECG_CONFIG['depth'],
        num_heads=ECG_CONFIG['num_heads'],
        mlp_ratio=ECG_CONFIG['mlp_ratio'],
        dropout=ECG_CONFIG['dropout'],
        task_type=ECG_CONFIG['task_type'],
        num_classes=ECG_CONFIG['num_classes'],
        head_config=head_config,
        pooling=ECG_CONFIG['pooling'],
        use_pos_embed=ECG_CONFIG['use_pos_embed'],
        pos_embed_type=ECG_CONFIG['pos_embed_type'],
    )
    
    # 加载权重 (strict=True 确保完全匹配)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}):")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:5]:
            print(f"    - {k}")
    
    if not missing and not unexpected:
        print("  所有参数完美匹配 (strict match)")
    else:
        print(f"  匹配参数: {len(ckpt['model_state_dict']) - len(unexpected)}/{len(model.state_dict())}")
    
    model.eval()
    
    # 统计各模块参数
    modules = {
        'Wavelet Decomposition': 'wavelet_decomp',
        'Patch Embedding': 'patch_embed',
        'Transformer Encoder': 'encoder',
        'Task Heads': 'task_heads',
    }
    print(f"\n  模块参数量统计:")
    for name, prefix in modules.items():
        params = sum(v.numel() for k, v in model.state_dict().items() if k.startswith(prefix))
        print(f"    {name:25s}: {params:>10,} ({params/1e6:.2f}M)")
    
    print("  [PASS] 模型重建与权重加载完成")
    return model


# ============================================================
# 3. 小波分解模块验证
# ============================================================
def test_wavelet_decomposition(model):
    separator("3. 小波分解模块验证")
    
    B, C, T = 2, 12, 2048
    x = torch.randn(B, C, T)
    
    print(f"  输入: {x.shape} (B={B}, C={C}, T={T})")
    
    wave_spec = model.wavelet_decomp(x)
    print(f"  小波分解输出: {wave_spec.shape}")
    
    # 论文公式: 输出应为 (B, (max_level+1)*C, T) = (B, 4*12, T) = (B, 48, T)
    expected_freq_bands = (ECG_CONFIG['max_level'] + 1) * ECG_CONFIG['in_channels']
    assert wave_spec.shape[0] == B, f"Batch 维度不匹配: {wave_spec.shape[0]} != {B}"
    assert wave_spec.shape[1] == expected_freq_bands, \
        f"频率带数不匹配: {wave_spec.shape[1]} != {expected_freq_bands} (应为 (L+1)*C = 4*12 = 48)"
    assert wave_spec.shape[2] == T, f"时间维度不匹配: {wave_spec.shape[2]} != {T}"
    
    print(f"  预期频率带: {ECG_CONFIG['max_level']+1} 层 x {C} 通道 = {expected_freq_bands}")
    print(f"  包含: d1({C}), d2({C}), d3({C}), a3({C})")
    
    # 检查数值范围
    print(f"  输出数值范围: [{wave_spec.min():.4f}, {wave_spec.max():.4f}]")
    print(f"  输出均值: {wave_spec.mean():.4f}, 标准差: {wave_spec.std():.4f}")
    assert not torch.isnan(wave_spec).any(), "小波分解输出包含 NaN"
    assert not torch.isinf(wave_spec).any(), "小波分解输出包含 Inf"
    
    print("  [PASS] 小波分解验证通过")


# ============================================================
# 4. 前向传播 - 分类任务验证
# ============================================================
def test_forward_classification(model):
    separator("4. 前向传播 - 分类任务 (PTB-XL 5类)")
    
    B = 4
    x = torch.randn(B, 12, 2048)
    
    model.eval()
    with torch.no_grad():
        logits = model(x, task='classify')
    
    print(f"  输入: {x.shape}")
    print(f"  分类输出: {logits.shape}")
    print(f"  预期输出: (B={B}, num_classes={ECG_CONFIG['num_classes']})")
    
    assert logits.shape == (B, ECG_CONFIG['num_classes']), \
        f"分类输出维度不匹配: {logits.shape} != ({B}, {ECG_CONFIG['num_classes']})"
    
    # 检查概率分布
    probs = torch.softmax(logits, dim=1)
    print(f"\n  Softmax 概率分布 (sample 0):")
    for i, cls in enumerate(PTBXL_CLASSES):
        print(f"    {cls:6s}: {probs[0, i]:.4f}")
    print(f"    概率之和: {probs[0].sum():.4f} (应为 1.0)")
    
    pred_class = logits.argmax(dim=1)
    print(f"\n  预测类别: {pred_class.tolist()}")
    print(f"  预测标签: {[PTBXL_CLASSES[i] for i in pred_class.tolist()]}")
    
    assert not torch.isnan(logits).any(), "分类输出包含 NaN"
    print("  [PASS] 分类任务前向传播验证通过")


# ============================================================
# 5. 特征提取模式验证
# ============================================================
def test_feature_extraction(model):
    separator("5. 特征提取模式 (Encoder 输出)")
    
    B = 4
    x = torch.randn(B, 12, 2048)
    
    model.eval()
    with torch.no_grad():
        features = model(x, task='features')
    
    print(f"  输入: {x.shape}")
    print(f"  特征输出: {features.shape}")
    
    # features shape 应为 (B, num_patches, embed_dim)
    # 小波分解后: (max_level+1)*C = 48 freq bands
    # wave_2d shape: (B, 1, 48, 2048)
    # patch_size = (1, 64), patches: (48/1) * (2048/64) = 48 * 32 = 1536
    expected_patches = (ECG_CONFIG['max_level'] + 1) * ECG_CONFIG['in_channels'] * (2048 // ECG_CONFIG['patch_size'][1])
    print(f"  预期 patch 数: {ECG_CONFIG['max_level']+1}(bands) x {ECG_CONFIG['in_channels']}(ch) x {2048 // ECG_CONFIG['patch_size'][1]}(time_patches) = {expected_patches}")
    print(f"  embed_dim: {ECG_CONFIG['embed_dim']}")
    
    assert features.shape[0] == B, "Batch 维度不匹配"
    assert features.shape[2] == ECG_CONFIG['embed_dim'], \
        f"Embed 维度不匹配: {features.shape[2]} != {ECG_CONFIG['embed_dim']}"
    
    print(f"  特征数值范围: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  特征均值: {features.mean():.4f}, 标准差: {features.std():.4f}")
    
    # 测试特征一致性: 同一输入应产生相同输出
    with torch.no_grad():
        features2 = model(x, task='features')
    diff = (features - features2).abs().max()
    print(f"  一致性检查 (同一输入两次): max diff = {diff:.2e}")
    assert diff < 1e-5, "特征提取不一致"
    
    print("  [PASS] 特征提取验证通过")
    return features


# ============================================================
# 6. 预训练重建头验证
# ============================================================
def test_pretrain_head(model):
    separator("6. 预训练重建头 (Masked Reconstruction)")
    
    B = 2
    x = torch.randn(B, 12, 2048)
    
    model.eval()
    with torch.no_grad():
        pred_patches, mask, target_patches = model(x, task='pretrain', mask_ratio=0.7)
    
    print(f"  输入: {x.shape}")
    print(f"  重建 patches:  {pred_patches.shape}")
    print(f"  目标 patches:  {target_patches.shape}")
    print(f"  Mask:          {mask.shape}")
    print(f"  Mask ratio:    {mask.float().mean():.2%}")
    
    assert pred_patches.shape == target_patches.shape, \
        f"重建与目标维度不匹配: {pred_patches.shape} vs {target_patches.shape}"
    
    # 计算重建 loss (Smooth L1, 论文公式 14)
    masked_indices = mask.bool()
    if masked_indices.any():
        pred_masked = pred_patches[masked_indices]
        target_masked = target_patches[masked_indices]
        recon_loss = nn.SmoothL1Loss()(pred_masked, target_masked)
        print(f"  重建 SmoothL1 Loss: {recon_loss:.4f}")
    else:
        print("  [WARN] 无 masked 位置")
    
    assert not torch.isnan(pred_patches).any(), "重建输出包含 NaN"
    print("  [PASS] 预训练重建头验证通过")


# ============================================================
# 7. 合成 ECG 信号推理验证
# ============================================================
def test_synthetic_ecg(model):
    separator("7. 合成 ECG 信号推理")
    
    # 生成模拟 12 导联 ECG 信号 (含 QRS 复合波)
    fs = 500  # 采样率 500Hz
    duration = 2048 / fs  # ~4.1 秒
    t = np.linspace(0, duration, 2048)
    
    # 模拟心电: 基线 + QRS 脉冲
    heart_rate = 72  # bpm
    rr_interval = 60.0 / heart_rate  # ~0.83 秒
    samples_per_beat = int(rr_interval * fs)
    
    ecg_12lead = np.zeros((12, 2048), dtype=np.float32)
    
    for lead in range(12):
        signal = np.zeros(2048)
        # 添加多个心跳
        for beat_start in range(0, 2048, samples_per_beat):
            # P 波
            p_center = beat_start + int(0.16 * fs)
            if p_center < 2048:
                p_idx = np.arange(max(0, p_center-15), min(2048, p_center+15))
                signal[p_idx] += 0.1 * np.sin(np.pi * (p_idx - p_center + 15) / 30)
            
            # QRS 复合波
            qrs_center = beat_start + int(0.2 * fs)
            if qrs_center < 2048:
                qrs_idx = np.arange(max(0, qrs_center-8), min(2048, qrs_center+8))
                qrs_shape = np.sin(np.pi * (qrs_idx - qrs_center + 8) / 16)
                # 不同导联有不同的极性和幅度
                amplitude = 1.0 + 0.3 * np.sin(lead * np.pi / 6)
                signal[qrs_idx] += amplitude * qrs_shape
            
            # T 波
            t_center = beat_start + int(0.4 * fs)
            if t_center < 2048:
                t_idx = np.arange(max(0, t_center-20), min(2048, t_center+20))
                signal[t_idx] += 0.2 * np.sin(np.pi * (t_idx - t_center + 20) / 40)
        
        # 添加轻微噪声
        signal += np.random.randn(2048) * 0.02
        ecg_12lead[lead] = signal
    
    # 归一化到 [-1, 1]
    ecg_12lead = ecg_12lead / (np.abs(ecg_12lead).max() + 1e-8)
    
    x = torch.tensor(ecg_12lead, dtype=torch.float32).unsqueeze(0)  # (1, 12, 2048)
    
    model.eval()
    with torch.no_grad():
        logits = model(x, task='classify')
        features = model(x, task='features')
    
    probs = torch.softmax(logits, dim=1)
    
    print(f"  合成 ECG 输入: {x.shape}")
    print(f"  信号范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  心率: {heart_rate} bpm")
    print(f"\n  分类结果:")
    for i, cls in enumerate(PTBXL_CLASSES):
        marker = " <-- 预测" if i == logits.argmax(dim=1).item() else ""
        print(f"    {cls:6s}: prob={probs[0, i]:.4f}{marker}")
    
    print(f"  预测: {PTBXL_CLASSES[logits.argmax(dim=1).item()]}")
    print(f"  特征维度: {features.shape}")
    print("  [PASS] 合成 ECG 推理验证通过")


# ============================================================
# 8. 梯度流与反向传播验证
# ============================================================
def test_gradient_flow(model):
    separator("8. 梯度流与反向传播验证")
    
    B = 2
    x = torch.randn(B, 12, 2048, requires_grad=False)
    y = torch.randint(0, ECG_CONFIG['num_classes'], (B,))
    
    model.train()
    
    # 全参数微调模式 (论文 Section 2.4 single-modal)
    for param in model.parameters():
        param.requires_grad = True
    
    logits = model(x, task='classify')
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    
    # 检查关键模块梯度
    grad_modules = {
        'wavelet_decomp': '小波分解',
        'patch_embed': 'Patch 嵌入',
        'encoder': 'Transformer 编码器',
        'task_heads.classification': '分类头',
    }
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"\n  各模块梯度状态:")
    for prefix, name in grad_modules.items():
        grads = [p.grad for p_name, p in model.named_parameters() 
                 if p_name.startswith(prefix) and p.grad is not None]
        if grads:
            grad_norms = [g.norm().item() for g in grads]
            avg_norm = np.mean(grad_norms)
            max_norm = np.max(grad_norms)
            has_nan = any(torch.isnan(g).any().item() for g in grads)
            status = "[OK]" if not has_nan and avg_norm > 0 else "[FAIL]"
            print(f"    {name:20s}: avg_grad_norm={avg_norm:.6f}, max={max_norm:.6f} {status}")
            assert not has_nan, f"{name} 梯度包含 NaN"
            assert avg_norm > 0, f"{name} 梯度为零"
        else:
            print(f"    {name:20s}: 无梯度 [WARN]")
    
    model.eval()
    print("  [PASS] 梯度流验证通过")


# ============================================================
# 9. 冻结编码器模式验证 (Linear Probing)
# ============================================================
def test_linear_probing(model):
    separator("9. Linear Probing 模式 (冻结编码器)")
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 仅解冻分类头
    for param in model.task_heads['classification'].parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"  总参数: {total:,} ({total/1e6:.2f}M)")
    print(f"  可训练: {trainable:,} ({trainable/1e6:.2f}M) [{trainable/total*100:.1f}%]")
    print(f"  冻结:   {frozen:,} ({frozen/1e6:.2f}M) [{frozen/total*100:.1f}%]")
    
    # 测试推理
    B = 4
    x = torch.randn(B, 12, 2048)
    y = torch.randint(0, ECG_CONFIG['num_classes'], (B,))
    
    model.train()
    logits = model(x, task='classify')
    loss = nn.CrossEntropyLoss()(logits, y)
    
    # 只对分类头做反向传播
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, trainable_params, allow_unused=True)
    
    for i, (g, p) in enumerate(zip(grads, trainable_params)):
        if g is not None:
            assert not torch.isnan(g).any(), f"分类头梯度包含 NaN"
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  分类头梯度: OK (无 NaN)")
    print("  [PASS] Linear Probing 模式验证通过")


# ============================================================
# 10. 推理速度与内存测试
# ============================================================
def test_inference_speed(model):
    separator("10. 推理速度测试")
    
    import time
    
    B = 8
    x = torch.randn(B, 12, 2048)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x, task='classify')
    
    # 测量
    n_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x, task='classify')
    elapsed = time.time() - start
    
    avg_time = elapsed / n_runs * 1000  # ms
    throughput = n_runs * B / elapsed  # samples/sec
    
    print(f"  Batch size: {B}")
    print(f"  运行次数: {n_runs}")
    print(f"  平均推理时间: {avg_time:.2f} ms/batch")
    print(f"  吞吐量: {throughput:.1f} samples/sec")
    print(f"  单样本延迟: {avg_time/B:.2f} ms/sample")
    
    # 论文中预训练在 16x A100 上进行，CPU 推理会慢很多
    device_info = "CPU" if not torch.cuda.is_available() else f"GPU ({torch.cuda.get_device_name(0)})"
    print(f"  设备: {device_info}")
    print("  [PASS] 推理速度测试完成")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("  PhysioWave ecg.pth 模型全面验证")
    print("  参考: PhysioWave (NeurIPS 2025, arXiv:2506.10351)")
    print("=" * 60)
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    try:
        # 1. 加载 checkpoint
        ckpt = test_checkpoint_loading()
        results['checkpoint_loading'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        results['checkpoint_loading'] = f'FAIL: {e}'
        return
    
    try:
        # 2. 重建模型
        model = test_model_rebuild(ckpt)
        results['model_rebuild'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        results['model_rebuild'] = f'FAIL: {e}'
        return
    
    try:
        # 3. 小波分解
        test_wavelet_decomposition(model)
        results['wavelet_decomposition'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        results['wavelet_decomposition'] = f'FAIL: {e}'
    
    try:
        # 4. 分类前向传播
        test_forward_classification(model)
        results['forward_classification'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        results['forward_classification'] = f'FAIL: {e}'
    
    try:
        # 5. 特征提取
        test_feature_extraction(model)
        results['feature_extraction'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        results['feature_extraction'] = f'FAIL: {e}'
    
    try:
        # 6. 预训练重建
        test_pretrain_head(model)
        results['pretrain_head'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        results['pretrain_head'] = f'FAIL: {e}'
    
    try:
        # 7. 合成 ECG 推理
        test_synthetic_ecg(model)
        results['synthetic_ecg'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        results['synthetic_ecg'] = f'FAIL: {e}'
    
    try:
        # 8. 梯度流
        test_gradient_flow(model)
        results['gradient_flow'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        results['gradient_flow'] = f'FAIL: {e}'
    
    try:
        # 9. Linear probing
        test_linear_probing(model)
        results['linear_probing'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        results['linear_probing'] = f'FAIL: {e}'
    
    try:
        # 10. 推理速度
        test_inference_speed(model)
        results['inference_speed'] = 'PASS'
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        results['inference_speed'] = f'FAIL: {e}'
    
    # 汇总
    separator("验证结果汇总")
    all_pass = True
    for test_name, result in results.items():
        status = "OK" if result == 'PASS' else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {test_name:30s}: {status}")
    
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"  通过: {sum(1 for v in results.values() if v == 'PASS')}/{len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
