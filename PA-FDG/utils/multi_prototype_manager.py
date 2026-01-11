#!/usr/bin/env python3
"""
多原型管理器
每个类用K个原型表示，更好地捕捉类内多样性
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


class MultiPrototypeManager(nn.Module):
    """
    多原型网络：每个类K个原型
    
    核心思想:
    - 单原型无法捕捉类内多样性
    - Preictal有早/中/晚期，模式不同
    - 多原型可以更精细地表示类别
    
    预期效果:
    - Spec提升 +8%
    - 更精细的决策边界
    """
    
    def __init__(self, feature_dim=128, n_prototypes=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_prototypes = n_prototypes
        
        # 每个类K个原型 (可学习参数)
        self.proto_preictal = nn.Parameter(
            torch.randn(n_prototypes, feature_dim)
        )
        self.proto_interictal = nn.Parameter(
            torch.randn(n_prototypes, feature_dim)
        )
        
        print(f"✅ 多原型管理器初始化: 每类{n_prototypes}个原型")
    
    def forward(self, features):
        """
        基于多原型计算logits
        
        Args:
            features: (N, feature_dim)
        
        Returns:
            logits: (N, 2)
        """
        # 计算到所有preictal原型的距离
        dist_preictal = torch.cdist(
            features, self.proto_preictal
        )  # (N, K)
        
        # 计算到所有interictal原型的距离
        dist_interictal = torch.cdist(
            features, self.proto_interictal
        )  # (N, K)
        
        # 使用最小距离（最近的原型）
        min_dist_preictal = dist_preictal.min(dim=1)[0]  # (N,)
        min_dist_interictal = dist_interictal.min(dim=1)[0]  # (N,)
        
        # 转换为logits (距离越小，logit越大)
        logits = torch.stack([
            -min_dist_interictal,
            -min_dist_preictal
        ], dim=1)  # (N, 2)
        
        return logits
    
    def initialize_prototypes_kmeans(self, features, labels):
        """
        使用K-means初始化原型
        
        Args:
            features: (N, feature_dim) tensor
            labels: (N,) tensor
        """
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        for label in [0, 1]:
            class_features = features_np[labels_np == label]
            
            if len(class_features) < self.n_prototypes:
                print(f"⚠️  类别{label}样本数({len(class_features)})少于原型数({self.n_prototypes})")
                continue
            
            # K-means聚类
            kmeans = KMeans(
                n_clusters=self.n_prototypes,
                random_state=42,
                n_init=10,
                max_iter=100
            )
            kmeans.fit(class_features)
            
            # 更新原型
            centers = torch.tensor(
                kmeans.cluster_centers_,
                dtype=torch.float32,
                device=features.device
            )
            
            if label == 0:
                self.proto_interictal.data = centers
            else:
                self.proto_preictal.data = centers
        
        print(f"✅ K-means初始化完成")
    
    def get_prototype_assignments(self, features, labels):
        """
        获取每个样本最近的原型索引
        用于分析和可视化
        
        Args:
            features: (N, feature_dim)
            labels: (N,)
        
        Returns:
            assignments: (N,) 每个样本对应的原型索引
        """
        assignments = torch.zeros(len(features), dtype=torch.long)
        
        for i, (feat, label) in enumerate(zip(features, labels)):
            if label == 1:  # preictal
                dists = torch.norm(
                    feat.unsqueeze(0) - self.proto_preictal, dim=1
                )
            else:  # interictal
                dists = torch.norm(
                    feat.unsqueeze(0) - self.proto_interictal, dim=1
                )
            assignments[i] = dists.argmin()
        
        return assignments
    
    def compute_prototype_quality(self, features, labels):
        """
        计算原型质量指标
        
        Returns:
            dict with:
            - intra_class_dist: 类内平均距离
            - inter_class_dist: 类间平均距离
            - separation_ratio: 分离度 (inter/intra)
        """
        with torch.no_grad():
            # 类内距离
            intra_dist_pre = 0
            intra_dist_int = 0
            
            pre_features = features[labels == 1]
            int_features = features[labels == 0]
            
            if len(pre_features) > 0:
                dist_pre = torch.cdist(pre_features, self.proto_preictal)
                intra_dist_pre = dist_pre.min(dim=1)[0].mean().item()
            
            if len(int_features) > 0:
                dist_int = torch.cdist(int_features, self.proto_interictal)
                intra_dist_int = dist_int.min(dim=1)[0].mean().item()
            
            intra_dist = (intra_dist_pre + intra_dist_int) / 2
            
            # 类间距离
            inter_dist = torch.cdist(
                self.proto_preictal, self.proto_interictal
            ).min().item()
            
            # 分离度
            separation_ratio = inter_dist / (intra_dist + 1e-7)
            
            return {
                'intra_class_dist': intra_dist,
                'inter_class_dist': inter_dist,
                'separation_ratio': separation_ratio
            }


def test_multi_prototype():
    """测试多原型管理器"""
    print("\n" + "="*70)
    print("测试多原型管理器")
    print("="*70)
    
    # 创建模拟数据
    n_samples = 200
    feature_dim = 128
    
    # 模拟3个子类的preictal
    features_pre1 = torch.randn(50, feature_dim) + torch.tensor([2.0] * feature_dim)
    features_pre2 = torch.randn(50, feature_dim) + torch.tensor([0.0] * feature_dim)
    features_pre3 = torch.randn(50, feature_dim) + torch.tensor([-2.0] * feature_dim)
    
    # 模拟interictal
    features_int = torch.randn(50, feature_dim) + torch.tensor([5.0] * feature_dim)
    
    features = torch.cat([features_pre1, features_pre2, features_pre3, features_int])
    labels = torch.cat([
        torch.ones(150),
        torch.zeros(50)
    ]).long()
    
    # 创建多原型管理器
    manager = MultiPrototypeManager(feature_dim=feature_dim, n_prototypes=3)
    
    # K-means初始化
    manager.initialize_prototypes_kmeans(features, labels)
    
    # 前向传播
    logits = manager(features)
    probs = torch.softmax(logits, dim=1)
    
    # 计算准确率
    preds = probs.argmax(dim=1)
    acc = (preds == labels).float().mean()
    
    print(f"\n准确率: {acc:.4f}")
    
    # 原型质量
    quality = manager.compute_prototype_quality(features, labels)
    print(f"\n原型质量:")
    print(f"  类内距离: {quality['intra_class_dist']:.4f}")
    print(f"  类间距离: {quality['inter_class_dist']:.4f}")
    print(f"  分离度: {quality['separation_ratio']:.4f}")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test_multi_prototype()
