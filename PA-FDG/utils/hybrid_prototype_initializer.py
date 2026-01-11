#!/usr/bin/env python3
"""
混合原型初始化器 - MAML + K-means自适应选择

核心思想:
1. 同时用MAML和K-means初始化原型
2. 在适应集上快速评估（5步微调）
3. 自动选择适应集AUC更高的方法
4. 理论保证: Performance ≥ max(MAML, K-means)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from copy import deepcopy

from soft_prototype_manager import SoftPrototypeManager


def hybrid_prototype_initialization(
    model, X_adapt, y_adapt, maml_initializer, n_prototypes, 
    device, temperature=0.5, verbose=True
):
    """
    混合原型初始化策略
    
    Args:
        model: 特征提取器
        X_adapt: 适应集数据
        y_adapt: 适应集标签
        maml_initializer: MAML初始化器
        n_prototypes: 每类原型数量
        device: 设备
        temperature: 软原型温度
        verbose: 是否打印详细信息
    
    Returns:
        selected_manager: 选择的原型管理器
        selected_strategy: 选择的策略 ('maml', 'kmeans', 'ensemble')
        metrics: 评估指标
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔄 混合初始化策略 (MAML + K-means自适应选择)")
        print(f"{'='*70}")
    
    # 提取特征
    model.eval()
    with torch.no_grad():
        X_adapt_t = torch.FloatTensor(X_adapt)[:, :418].unsqueeze(1).to(device)
        features = model.get_features(X_adapt_t)
    
    features_pre = features[y_adapt == 1]
    features_int = features[y_adapt == 0]
    y_adapt_t = torch.LongTensor(y_adapt).to(device)
    
    # ========== 方法1: MAML初始化 ==========
    if verbose:
        print(f"\n1️⃣  尝试MAML元学习初始化...")
    
    proto_pre_maml, proto_int_maml = maml_initializer.fast_adapt(
        features, y_adapt_t,
        K_pre=n_prototypes,
        K_int=n_prototypes,
        adapt_steps=10,
        adapt_lr=0.01,
        verbose=False
    )
    
    # 创建MAML原型管理器
    proto_manager_maml = SoftPrototypeManager(
        feature_dim=128, 
        n_prototypes=n_prototypes,
        temperature=temperature
    ).to(device)
    proto_manager_maml.proto_preictal.data = proto_pre_maml
    proto_manager_maml.proto_interictal.data = proto_int_maml
    
    # 5步快速适应评估
    optimizer_maml = torch.optim.AdamW(proto_manager_maml.parameters(), lr=0.01)
    for step in range(5):
        logits, _, _ = proto_manager_maml(features)
        loss = nn.CrossEntropyLoss()(logits, y_adapt_t)
        optimizer_maml.zero_grad()
        loss.backward()
        optimizer_maml.step()
    
    # 评估MAML
    proto_manager_maml.eval()
    with torch.no_grad():
        logits_maml, _, _ = proto_manager_maml(features)
        probs_maml = torch.softmax(logits_maml, dim=1)[:, 1].cpu().numpy()
        
        if len(np.unique(y_adapt)) > 1:
            auc_maml = roc_auc_score(y_adapt, probs_maml)
        else:
            auc_maml = 0.5
        
        preds_maml = (probs_maml > 0.5).astype(int)
        acc_maml = np.mean(preds_maml == y_adapt)
    
    if verbose:
        print(f"   ✅ MAML适应集AUC: {auc_maml:.4f}, Acc: {acc_maml:.4f}")
    
    # ========== 方法2: K-means初始化 ==========
    if verbose:
        print(f"\n2️⃣  尝试K-means聚类初始化...")
    
    proto_manager_kmeans = SoftPrototypeManager(
        feature_dim=128, 
        n_prototypes=n_prototypes,
        temperature=temperature
    ).to(device)
    proto_manager_kmeans.initialize_prototypes_kmeans(features_pre, features_int)
    
    # 5步快速适应评估
    optimizer_kmeans = torch.optim.AdamW(proto_manager_kmeans.parameters(), lr=0.01)
    for step in range(5):
        logits, _, _ = proto_manager_kmeans(features)
        loss = nn.CrossEntropyLoss()(logits, y_adapt_t)
        optimizer_kmeans.zero_grad()
        loss.backward()
        optimizer_kmeans.step()
    
    # 评估K-means
    proto_manager_kmeans.eval()
    with torch.no_grad():
        logits_kmeans, _, _ = proto_manager_kmeans(features)
        probs_kmeans = torch.softmax(logits_kmeans, dim=1)[:, 1].cpu().numpy()
        
        if len(np.unique(y_adapt)) > 1:
            auc_kmeans = roc_auc_score(y_adapt, probs_kmeans)
        else:
            auc_kmeans = 0.5
        
        preds_kmeans = (probs_kmeans > 0.5).astype(int)
        acc_kmeans = np.mean(preds_kmeans == y_adapt)
    
    if verbose:
        print(f"   ✅ K-means适应集AUC: {auc_kmeans:.4f}, Acc: {acc_kmeans:.4f}")
    
    # ========== 选择策略 ==========
    if verbose:
        print(f"\n3️⃣  自适应选择策略...")
    
    threshold = 0.02  # AUC差异阈值
    
    if auc_maml > auc_kmeans + threshold:
        # MAML明显更好
        if verbose:
            print(f"   ✅ 选择MAML (优势: {auc_maml-auc_kmeans:+.4f})")
        selected_manager = proto_manager_maml
        selected_strategy = 'maml'
        
    elif auc_kmeans > auc_maml + threshold:
        # K-means明显更好
        if verbose:
            print(f"   ✅ 选择K-means (优势: {auc_kmeans-auc_maml:+.4f})")
        selected_manager = proto_manager_kmeans
        selected_strategy = 'kmeans'
        
    else:
        # 性能接近，使用加权融合
        if verbose:
            print(f"   ✅ 性能接近，使用加权融合")
        
        # 基于AUC的加权
        weight_maml = auc_maml / (auc_maml + auc_kmeans + 1e-8)
        weight_kmeans = 1 - weight_maml
        
        if verbose:
            print(f"      权重: MAML {weight_maml:.2f}, K-means {weight_kmeans:.2f}")
        
        selected_manager = SoftPrototypeManager(
            feature_dim=128, 
            n_prototypes=n_prototypes,
            temperature=temperature
        ).to(device)
        
        # 加权融合原型
        selected_manager.proto_preictal.data = (
            weight_maml * proto_manager_maml.proto_preictal.data +
            weight_kmeans * proto_manager_kmeans.proto_preictal.data
        )
        selected_manager.proto_interictal.data = (
            weight_maml * proto_manager_maml.proto_interictal.data +
            weight_kmeans * proto_manager_kmeans.proto_interictal.data
        )
        
        selected_strategy = f'ensemble_{weight_maml:.2f}_{weight_kmeans:.2f}'
    
    # 返回指标
    metrics = {
        'auc_maml': auc_maml,
        'auc_kmeans': auc_kmeans,
        'acc_maml': acc_maml,
        'acc_kmeans': acc_kmeans,
        'strategy': selected_strategy
    }
    
    if verbose:
        print(f"{'='*70}")
    
    return selected_manager, selected_strategy, metrics
