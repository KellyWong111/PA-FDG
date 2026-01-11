#!/usr/bin/env python3
"""
软原型管理器 - v7核心组件

核心改进: 用加权距离代替最小距离
- v5硬匹配: min_dist = dist.min(dim=1)[0]  (只看最近的1个)
- v7软匹配: weighted_dist = (weights * dist).sum(dim=1)  (考虑所有原型)

优势:
1. 更鲁棒 (不依赖单个原型)
2. 更平滑的决策边界
3. 自动学习原型重要性
4. 减少Sens损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class SoftPrototypeManager(nn.Module):
    """
    软原型网络: 使用加权距离而非最小距离
    
    核心思想:
    - 计算样本到所有原型的距离
    - 用softmax转换为权重
    - 加权求和得到软距离
    """
    def __init__(self, feature_dim=128, n_prototypes=3, temperature=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        
        # 可学习的原型
        self.proto_preictal = nn.Parameter(
            torch.randn(n_prototypes, feature_dim) * 0.1
        )
        self.proto_interictal = nn.Parameter(
            torch.randn(n_prototypes, feature_dim) * 0.1
        )
        
        print(f"✅ 软原型管理器初始化: 每类{n_prototypes}个原型, temperature={temperature}")
    
    def initialize_prototypes_kmeans(self, features_pre, features_int):
        """
        使用K-means初始化原型
        """
        with torch.no_grad():
            # Preictal原型 (使用 K-means++ 初始化)
            if len(features_pre) >= self.n_prototypes:
                kmeans_pre = KMeans(
                    n_clusters=self.n_prototypes, 
                    init='k-means++',  # 改进: 使用 K-means++
                    random_state=42, 
                    n_init=10
                )
                kmeans_pre.fit(features_pre.cpu().numpy())
                self.proto_preictal.data = torch.FloatTensor(
                    kmeans_pre.cluster_centers_
                ).to(self.proto_preictal.device)
            else:
                # 样本不够，随机选择
                indices = torch.randperm(len(features_pre))[:self.n_prototypes]
                self.proto_preictal.data = features_pre[indices]
            
            # Interictal原型 (使用 K-means++ 初始化)
            if len(features_int) >= self.n_prototypes:
                kmeans_int = KMeans(
                    n_clusters=self.n_prototypes, 
                    init='k-means++',  # 改进: 使用 K-means++
                    random_state=42, 
                    n_init=10
                )
                kmeans_int.fit(features_int.cpu().numpy())
                self.proto_interictal.data = torch.FloatTensor(
                    kmeans_int.cluster_centers_
                ).to(self.proto_interictal.device)
            else:
                indices = torch.randperm(len(features_int))[:self.n_prototypes]
                self.proto_interictal.data = features_int[indices]
        
        print(f"✅ K-means初始化完成")
    
    def forward(self, features):
        """
        软原型匹配
        
        Args:
            features: (N, feature_dim)
        
        Returns:
            logits: (N, 2)
            weights_pre: (N, n_prototypes) - 每个样本对preictal原型的权重
            weights_int: (N, n_prototypes) - 每个样本对interictal原型的权重
        """
        # 计算距离 (N, n_prototypes)
        dist_preictal = torch.cdist(features, self.proto_preictal)
        dist_interictal = torch.cdist(features, self.proto_interictal)
        
        # ===== 核心改进: 软加权 =====
        # 距离转换为相似度权重
        # 距离越小，权重越大
        weights_pre = F.softmax(-dist_preictal / self.temperature, dim=1)
        weights_int = F.softmax(-dist_interictal / self.temperature, dim=1)
        
        # 加权距离 (考虑所有原型)
        soft_dist_preictal = (weights_pre * dist_preictal).sum(dim=1)
        soft_dist_interictal = (weights_int * dist_interictal).sum(dim=1)
        
        # 转换为logits
        logits = torch.stack([
            -soft_dist_interictal,
            -soft_dist_preictal
        ], dim=1)
        
        return logits, weights_pre, weights_int
    
    def get_prototype_stats(self, features, labels):
        """
        获取原型统计信息（用于分析）
        """
        with torch.no_grad():
            features_pre = features[labels == 1]
            features_int = features[labels == 0]
            
            stats = {}
            
            # Preictal原型统计
            if len(features_pre) > 0:
                dist_pre = torch.cdist(features_pre, self.proto_preictal)
                weights_pre = F.softmax(-dist_pre / self.temperature, dim=1)
                
                stats['preictal'] = {
                    'avg_min_dist': dist_pre.min(dim=1)[0].mean().item(),
                    'avg_soft_dist': (weights_pre * dist_pre).sum(dim=1).mean().item(),
                    'weight_entropy': -(weights_pre * torch.log(weights_pre + 1e-8)).sum(dim=1).mean().item(),
                    'dominant_proto': weights_pre.argmax(dim=1).float().mean().item()
                }
            
            # Interictal原型统计
            if len(features_int) > 0:
                dist_int = torch.cdist(features_int, self.proto_interictal)
                weights_int = F.softmax(-dist_int / self.temperature, dim=1)
                
                stats['interictal'] = {
                    'avg_min_dist': dist_int.min(dim=1)[0].mean().item(),
                    'avg_soft_dist': (weights_int * dist_int).sum(dim=1).mean().item(),
                    'weight_entropy': -(weights_int * torch.log(weights_int + 1e-8)).sum(dim=1).mean().item(),
                    'dominant_proto': weights_int.argmax(dim=1).float().mean().item()
                }
            
            # 原型间距离
            proto_dist = torch.cdist(self.proto_preictal, self.proto_interictal)
            stats['inter_class_dist'] = proto_dist.mean().item()
            
            intra_dist_pre = torch.cdist(self.proto_preictal, self.proto_preictal)
            intra_dist_pre = intra_dist_pre[~torch.eye(self.n_prototypes, dtype=bool)]
            stats['intra_class_dist_pre'] = intra_dist_pre.mean().item()
            
            intra_dist_int = torch.cdist(self.proto_interictal, self.proto_interictal)
            intra_dist_int = intra_dist_int[~torch.eye(self.n_prototypes, dtype=bool)]
            stats['intra_class_dist_int'] = intra_dist_int.mean().item()
            
            return stats


class SoftPrototypeLoss(nn.Module):
    """
    软原型损失: 分类损失 + 原型质量损失
    """
    def __init__(self, lambda_proto=0.1):
        super().__init__()
        self.lambda_proto = lambda_proto
    
    def forward(self, logits, labels, proto_manager, features):
        """
        计算总损失
        
        Args:
            logits: 分类logits
            labels: 真实标签
            proto_manager: 软原型管理器
            features: 特征
        """
        # 1. 分类损失
        ce_loss = F.cross_entropy(logits, labels)
        
        # 2. 原型质量损失（可选）
        # 确保原型之间有足够距离
        proto_pre = proto_manager.proto_preictal
        proto_int = proto_manager.proto_interictal
        
        # 类内原型应该分散
        intra_dist_pre = torch.cdist(proto_pre, proto_pre)
        intra_dist_pre = intra_dist_pre[~torch.eye(proto_manager.n_prototypes, dtype=bool, device=proto_pre.device)]
        intra_loss_pre = F.relu(0.5 - intra_dist_pre).mean() if len(intra_dist_pre) > 0 else 0
        
        intra_dist_int = torch.cdist(proto_int, proto_int)
        intra_dist_int = intra_dist_int[~torch.eye(proto_manager.n_prototypes, dtype=bool, device=proto_int.device)]
        intra_loss_int = F.relu(0.5 - intra_dist_int).mean() if len(intra_dist_int) > 0 else 0
        
        proto_diversity_loss = intra_loss_pre + intra_loss_int
        
        # 总损失
        total_loss = ce_loss + self.lambda_proto * proto_diversity_loss
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'proto_diversity_loss': proto_diversity_loss.item() if isinstance(proto_diversity_loss, torch.Tensor) else proto_diversity_loss
        }
