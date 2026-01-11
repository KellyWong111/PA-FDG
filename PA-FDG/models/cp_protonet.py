#!/usr/bin/env python3
"""
CP-ProtoNet: Contrastive-Prototypical Network for Few-Shot Seizure Prediction

核心创新:
1. 对比预训练 (Contrastive Pretraining)
2. 原型网络 (Prototypical Networks)
3. 在线适应 (Online Adaptation)

目标: AUC 0.75-0.78
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import random

from .dstg_oml_model_v2 import DSTG_Model_V2


class TripletBuilder:
    """
    三元组构建器
    
    构建策略:
    - Anchor: 患者A的Preictal样本
    - Positive: 患者A的另一个Preictal样本 (同患者同类)
    - Negative: 患者B的Interictal样本 (不同患者不同类)
    
    目标: 学习判别性表示
    - 同患者同类样本接近
    - 不同患者不同类样本远离
    """
    
    def __init__(self, margin=1.0):
        self.margin = margin
    
    def create_triplets(self, X_list, y_list, patient_ids, num_triplets=1000):
        """
        创建三元组
        
        Args:
            X_list: list of arrays, 每个患者的数据
            y_list: list of arrays, 每个患者的标签
            patient_ids: list of str, 患者ID
            num_triplets: int, 要创建的三元组数量
        
        Returns:
            triplets: list of (anchor_idx, pos_idx, neg_idx, patient_a, patient_b)
        """
        triplets = []
        
        # 为每个患者建立索引
        patient_data = {}
        for i, pid in enumerate(patient_ids):
            patient_data[pid] = {
                'X': X_list[i],
                'y': y_list[i],
                'preictal_idx': np.where(y_list[i] == 1)[0],
                'interictal_idx': np.where(y_list[i] == 0)[0]
            }
        
        print(f"\n构建三元组 (目标: {num_triplets}个)...")
        
        # 创建三元组
        for _ in range(num_triplets):
            # 随机选择两个不同的患者
            patient_a, patient_b = random.sample(patient_ids, 2)
            
            # 患者A的Preictal样本
            if len(patient_data[patient_a]['preictal_idx']) < 2:
                continue
            
            # Anchor和Positive: 患者A的两个不同Preictal样本
            anchor_idx, pos_idx = random.sample(
                list(patient_data[patient_a]['preictal_idx']), 2
            )
            
            # Negative: 患者B的Interictal样本
            if len(patient_data[patient_b]['interictal_idx']) == 0:
                continue
            
            neg_idx = random.choice(patient_data[patient_b]['interictal_idx'])
            
            triplets.append({
                'anchor': patient_data[patient_a]['X'][anchor_idx],
                'positive': patient_data[patient_a]['X'][pos_idx],
                'negative': patient_data[patient_b]['X'][neg_idx],
                'patient_a': patient_a,
                'patient_b': patient_b
            })
        
        print(f"✅ 成功创建 {len(triplets)} 个三元组")
        return triplets
    
    def triplet_loss(self, anchor, positive, negative, margin=None):
        """
        三元组损失
        
        loss = max(0, ||anchor - positive||^2 - ||anchor - negative||^2 + margin)
        
        目标:
        - ||anchor - positive|| 尽可能小 (同类接近)
        - ||anchor - negative|| 尽可能大 (异类远离)
        """
        if margin is None:
            margin = self.margin
        
        # 计算距离
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        
        # 三元组损失
        loss = torch.relu(dist_pos - dist_neg + margin)
        
        return loss.mean()


class PrototypeManager:
    """
    原型管理器
    
    功能:
    1. 计算类别原型
    2. 基于原型距离预测
    3. 在线更新原型 (EMA)
    """
    
    def __init__(self, feature_dim=128, momentum=0.9):
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        # 原型
        self.proto_preictal = None
        self.proto_interictal = None
    
    def compute_prototypes(self, features, labels):
        """
        计算类别原型 (均值)
        
        Args:
            features: [N, feature_dim]
            labels: [N]
        
        Returns:
            proto_preictal: [feature_dim]
            proto_interictal: [feature_dim]
        """
        # Preictal原型
        preictal_features = features[labels == 1]
        if len(preictal_features) > 0:
            proto_preictal = preictal_features.mean(dim=0)
        else:
            proto_preictal = torch.zeros(self.feature_dim, device=features.device)
        
        # Interictal原型
        interictal_features = features[labels == 0]
        if len(interictal_features) > 0:
            proto_interictal = interictal_features.mean(dim=0)
        else:
            proto_interictal = torch.zeros(self.feature_dim, device=features.device)
        
        return proto_preictal, proto_interictal
    
    def predict_with_prototypes(self, features):
        """
        基于原型距离预测
        
        Args:
            features: [N, feature_dim]
        
        Returns:
            probs: [N, 2], 预测概率
            distances: [N, 2], 到两个原型的距离
        """
        # 计算到两个原型的距离
        dist_to_interictal = torch.norm(
            features - self.proto_interictal.unsqueeze(0), dim=1
        )
        dist_to_preictal = torch.norm(
            features - self.proto_preictal.unsqueeze(0), dim=1
        )
        
        # 距离越小，概率越大
        # 使用负距离作为logits
        logits = torch.stack([-dist_to_interictal, -dist_to_preictal], dim=1)
        probs = torch.softmax(logits, dim=1)
        
        distances = torch.stack([dist_to_interictal, dist_to_preictal], dim=1)
        
        return probs, distances
    
    def update_prototypes(self, new_features, new_labels):
        """
        在线更新原型 (指数移动平均)
        
        Args:
            new_features: [N, feature_dim]
            new_labels: [N]
        """
        # 计算新的原型
        new_proto_preictal, new_proto_interictal = self.compute_prototypes(
            new_features, new_labels
        )
        
        # EMA更新
        if self.proto_preictal is None:
            # 初始化
            self.proto_preictal = new_proto_preictal
            self.proto_interictal = new_proto_interictal
        else:
            # 指数移动平均
            self.proto_preictal = (
                self.momentum * self.proto_preictal + 
                (1 - self.momentum) * new_proto_preictal
            )
            self.proto_interictal = (
                self.momentum * self.proto_interictal + 
                (1 - self.momentum) * new_proto_interictal
            )


class CP_ProtoNet(nn.Module):
    """
    完整的CP-ProtoNet模型
    
    三阶段学习:
    1. 对比预训练 (17患者)
    2. 原型学习 (100样本)
    3. 在线适应 (持续更新)
    """
    
    def __init__(self, num_channels=22, feature_dim=19, 
                 hidden_dim=128, num_gcn_layers=3, num_heads=8,
                 use_hybrid_graph=False, fusion_mode='learned', fixed_alpha=0.6):
        super().__init__()
        
        print("="*70)
        print("初始化 CP-ProtoNet 模型")
        print("="*70)
        
        # 特征提取器 (DSTG)
        # 支持混合图
        if use_hybrid_graph:
            try:
                from dstg_oml_model_v2_hybrid import DSTGV2Hybrid
                self.feature_extractor = DSTGV2Hybrid(
                    num_channels=num_channels,
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    fusion_mode=fusion_mode,
                    fixed_alpha=fixed_alpha
                )
                print(f"✅ 使用混合图模型 (fusion_mode={fusion_mode})")
            except ImportError:
                print("⚠️ 混合图模块未找到，使用标准模型")
                self.feature_extractor = DSTG_Model_V2(
                    num_channels=num_channels,
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    num_gcn_layers=num_gcn_layers,
                    num_heads=num_heads
                )
        else:
            self.feature_extractor = DSTG_Model_V2(
                num_channels=num_channels,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_gcn_layers=num_gcn_layers,
                num_heads=num_heads
            )
        
        # 投影头 (用于对比学习)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # 原型管理器
        self.prototype_manager = PrototypeManager(
            feature_dim=hidden_dim,
            momentum=0.9
        )
        
        # 三元组构建器
        self.triplet_builder = TripletBuilder(margin=1.0)
        
        print("="*70)
        print("✅ CP-ProtoNet 初始化完成")
        print(f"特征维度: {hidden_dim}")
        print(f"投影维度: {hidden_dim // 4}")
        print("="*70)
    
    def get_features(self, x):
        """
        提取特征 (用于原型计算和预测)
        """
        return self.feature_extractor.get_features(x)
    
    def get_projected_features(self, x):
        """
        提取投影特征 (用于对比学习)
        """
        features = self.get_features(x)
        projected = self.projector(features)
        return projected
    
    def forward(self, x):
        """
        前向传播 (用于监督学习)
        """
        return self.feature_extractor(x)


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 测试 CP-ProtoNet 核心模块")
    print("="*70)
    
    # 1. 测试三元组构建
    print("\n1️⃣ 测试三元组构建:")
    
    # 模拟数据
    X_list = [np.random.randn(100, 418) for _ in range(3)]
    y_list = [np.random.randint(0, 2, 100) for _ in range(3)]
    patient_ids = ['chb01', 'chb02', 'chb03']
    
    triplet_builder = TripletBuilder()
    triplets = triplet_builder.create_triplets(
        X_list, y_list, patient_ids, num_triplets=50
    )
    print(f"   创建了 {len(triplets)} 个三元组")
    
    # 2. 测试三元组损失
    print("\n2️⃣ 测试三元组损失:")
    anchor = torch.randn(4, 128)
    positive = torch.randn(4, 128)
    negative = torch.randn(4, 128)
    
    loss = triplet_builder.triplet_loss(anchor, positive, negative)
    print(f"   三元组损失: {loss.item():.4f}")
    
    # 3. 测试原型管理器
    print("\n3️⃣ 测试原型管理器:")
    proto_manager = PrototypeManager(feature_dim=128)
    
    features = torch.randn(10, 128)
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    proto_preictal, proto_interictal = proto_manager.compute_prototypes(
        features, labels
    )
    print(f"   Preictal原型: {proto_preictal.shape}")
    print(f"   Interictal原型: {proto_interictal.shape}")
    
    # 4. 测试CP-ProtoNet模型
    print("\n4️⃣ 测试CP-ProtoNet模型:")
    model = CP_ProtoNet(
        num_channels=22,
        feature_dim=19,
        hidden_dim=128,
        num_gcn_layers=3,
        num_heads=8
    )
    
    # 测试输入
    x = torch.randn(4, 1, 418)
    
    # 提取特征
    features = model.get_features(x)
    print(f"   特征形状: {features.shape}")
    
    # 提取投影特征
    projected = model.get_projected_features(x)
    print(f"   投影特征形状: {projected.shape}")
    
    # 前向传播
    output = model(x)
    print(f"   输出形状: {output.shape}")
    
    print("\n✅ 所有测试通过！")
    print("="*70 + "\n")
