#!/usr/bin/env python3
"""
对比原型学习 - v8核心组件

核心思想: 显式优化原型质量
1. 类内紧凑: 样本靠近自己类的原型
2. 类间分离: 样本远离其他类的原型
3. 原型对比: 两类原型距离最大化

优势:
- 提升原型判别力
- 减少误判和漏检
- 与软原型协同
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastivePrototypeLoss(nn.Module):
    """
    对比原型损失
    
    三重优化目标:
    1. Intra-class compactness (类内紧凑)
    2. Inter-class separation (类间分离)
    3. Prototype contrast (原型对比)
    """
    def __init__(self, margin=1.0, lambda_intra=0.5, lambda_inter=0.3, lambda_proto=0.2):
        super().__init__()
        self.margin = margin
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.lambda_proto = lambda_proto
    
    def forward(self, features, labels, proto_pre, proto_int, use_soft=True, temperature=0.5):
        """
        计算对比原型损失
        
        Args:
            features: (N, feature_dim)
            labels: (N,)
            proto_pre: (K, feature_dim) - preictal原型
            proto_int: (K, feature_dim) - interictal原型
            use_soft: 是否使用软距离
            temperature: 软距离的温度参数
        """
        features_pre = features[labels == 1]
        features_int = features[labels == 0]
        
        # ===== 1. 类内紧凑损失 =====
        # 样本应该靠近自己类的原型
        intra_loss = 0
        
        if len(features_pre) > 0:
            dist_pre_to_proto = torch.cdist(features_pre, proto_pre)
            if use_soft:
                # 软距离: 加权平均
                weights = F.softmax(-dist_pre_to_proto / temperature, dim=1)
                intra_dist_pre = (weights * dist_pre_to_proto).sum(dim=1).mean()
            else:
                # 硬距离: 最小距离
                intra_dist_pre = dist_pre_to_proto.min(dim=1)[0].mean()
            intra_loss += intra_dist_pre
        
        if len(features_int) > 0:
            dist_int_to_proto = torch.cdist(features_int, proto_int)
            if use_soft:
                weights = F.softmax(-dist_int_to_proto / temperature, dim=1)
                intra_dist_int = (weights * dist_int_to_proto).sum(dim=1).mean()
            else:
                intra_dist_int = dist_int_to_proto.min(dim=1)[0].mean()
            intra_loss += intra_dist_int
        
        # ===== 2. 类间分离损失 =====
        # 样本应该远离其他类的原型 (margin-based)
        inter_loss = 0
        
        if len(features_pre) > 0:
            # preictal样本应该远离interictal原型
            dist_pre_to_int_proto = torch.cdist(features_pre, proto_int)
            if use_soft:
                weights = F.softmax(-dist_pre_to_int_proto / temperature, dim=1)
                inter_dist_pre = (weights * dist_pre_to_int_proto).sum(dim=1)
            else:
                inter_dist_pre = dist_pre_to_int_proto.min(dim=1)[0]
            
            # Margin loss: 希望距离 > margin
            inter_loss += F.relu(self.margin - inter_dist_pre).mean()
        
        if len(features_int) > 0:
            # interictal样本应该远离preictal原型
            dist_int_to_pre_proto = torch.cdist(features_int, proto_pre)
            if use_soft:
                weights = F.softmax(-dist_int_to_pre_proto / temperature, dim=1)
                inter_dist_int = (weights * dist_int_to_pre_proto).sum(dim=1)
            else:
                inter_dist_int = dist_int_to_pre_proto.min(dim=1)[0]
            
            inter_loss += F.relu(self.margin - inter_dist_int).mean()
        
        # ===== 3. 原型对比损失 =====
        # 两类原型之间的距离应该最大化
        proto_distances = torch.cdist(proto_pre, proto_int)
        proto_contrast_loss = -proto_distances.mean()  # 负号: 最大化距离
        
        # ===== 总损失 =====
        total_loss = (
            self.lambda_intra * intra_loss +
            self.lambda_inter * inter_loss +
            self.lambda_proto * proto_contrast_loss
        )
        
        # 统计信息
        stats = {
            'intra_loss': intra_loss.item() if isinstance(intra_loss, torch.Tensor) else intra_loss,
            'inter_loss': inter_loss.item() if isinstance(inter_loss, torch.Tensor) else inter_loss,
            'proto_contrast_loss': proto_contrast_loss.item(),
            'avg_proto_dist': proto_distances.mean().item()
        }
        
        return total_loss, stats


class CombinedLoss(nn.Module):
    """
    组合损失: 分类 + 对比原型
    """
    def __init__(self, 
                 lambda_ce=1.0,
                 lambda_contrastive=0.3,
                 margin=1.0,
                 lambda_intra=0.5,
                 lambda_inter=0.3,
                 lambda_proto=0.2):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_contrastive = lambda_contrastive
        
        self.contrastive_loss = ContrastivePrototypeLoss(
            margin=margin,
            lambda_intra=lambda_intra,
            lambda_inter=lambda_inter,
            lambda_proto=lambda_proto
        )
    
    def forward(self, logits, labels, features, proto_pre, proto_int, 
                use_soft=True, temperature=0.5):
        """
        计算组合损失
        
        Args:
            logits: 分类logits
            labels: 真实标签
            features: 特征
            proto_pre: preictal原型
            proto_int: interictal原型
            use_soft: 是否使用软距离
            temperature: 温度参数
        """
        # 1. 分类损失
        ce_loss = F.cross_entropy(logits, labels)
        
        # 2. 对比原型损失
        contrastive_loss, contrast_stats = self.contrastive_loss(
            features, labels, proto_pre, proto_int,
            use_soft=use_soft, temperature=temperature
        )
        
        # 3. 总损失
        total_loss = (
            self.lambda_ce * ce_loss +
            self.lambda_contrastive * contrastive_loss
        )
        
        # 统计信息
        stats = {
            'ce_loss': ce_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            **contrast_stats
        }
        
        return total_loss, stats
