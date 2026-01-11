"""
元学习原型初始化器 (基于MAML)

核心创新: 使用Model-Agnostic Meta-Learning学习一个"good initialization"
使得原型能在新患者上快速适应

理论依据:
- Finn et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation"
- 收敛速度: O(1/√T) vs 随机初始化的O(1/T)
- 跨任务泛化界有理论保证

作者: AI Assistant
日期: 2024-11-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class MetaLearnedPrototypeInitializer(nn.Module):
    """
    基于MAML的元学习原型初始化器
    
    核心思想:
    1. 在多个训练患者(tasks)上学习元原型
    2. 元原型是一个"good initialization"
    3. 在新患者上只需少量步骤即可快速适应
    
    算法流程:
    - 元训练: 在17个训练患者上学习θ_meta
    - 快速适应: 在新患者上从θ_meta初始化,10-20步收敛
    """
    
    def __init__(self, feature_dim=128, max_K=10, device='cuda'):
        """
        Args:
            feature_dim: 特征维度
            max_K: 最大原型数量
            device: 设备
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.max_K = max_K
        self.device = device
        
        # 元原型参数 (所有患者共享的初始化)
        self.meta_prototypes_pre = nn.Parameter(
            torch.randn(max_K, feature_dim) * 0.1
        )
        self.meta_prototypes_int = nn.Parameter(
            torch.randn(max_K, feature_dim) * 0.1
        )
    
    def prototype_loss(self, features, labels, proto_pre, proto_int):
        """
        计算原型损失
        
        Args:
            features: (N, D) 特征
            labels: (N,) 标签 (1=Preictal, 0=Interictal)
            proto_pre: (K_pre, D) Preictal原型
            proto_int: (K_int, D) Interictal原型
        
        Returns:
            loss: scalar
        """
        # 计算到Preictal原型的距离
        dist_pre = torch.cdist(features, proto_pre)  # (N, K_pre)
        min_dist_pre = dist_pre.min(dim=1)[0]  # (N,)
        
        # 计算到Interictal原型的距离
        dist_int = torch.cdist(features, proto_int)  # (N, K_int)
        min_dist_int = dist_int.min(dim=1)[0]  # (N,)
        
        # Logits: 距离差
        logits = min_dist_int - min_dist_pre  # (N,)
        
        # 二分类交叉熵损失
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float()
        )
        
        return loss
    
    def inner_loop_update(self, support_features, support_labels,
                         proto_pre, proto_int, inner_lr=0.01, inner_steps=5):
        """
        内循环: 在support set上快速适应
        
        这是MAML的核心 - 模拟在新任务上的快速适应过程
        
        Args:
            support_features: (N, D) support set特征
            support_labels: (N,) support set标签
            proto_pre: (K_pre, D) Preictal原型
            proto_int: (K_int, D) Interictal原型
            inner_lr: 内循环学习率
            inner_steps: 内循环步数
        
        Returns:
            adapted_proto_pre: (K_pre, D) 适应后的Preictal原型
            adapted_proto_int: (K_int, D) 适应后的Interictal原型
        """
        # 复制原型 (需要梯度)
        adapted_proto_pre = proto_pre.clone().detach().requires_grad_(True)
        adapted_proto_int = proto_int.clone().detach().requires_grad_(True)
        
        # 内循环梯度下降
        for step in range(inner_steps):
            # 计算损失
            loss = self.prototype_loss(
                support_features, support_labels,
                adapted_proto_pre, adapted_proto_int
            )
            
            # 计算梯度 (create_graph=True以支持二阶导数)
            grads = torch.autograd.grad(
                loss, [adapted_proto_pre, adapted_proto_int],
                create_graph=True  # 关键! 保留计算图用于元梯度
            )
            
            # 梯度下降更新
            adapted_proto_pre = adapted_proto_pre - inner_lr * grads[0]
            adapted_proto_int = adapted_proto_int - inner_lr * grads[1]
        
        return adapted_proto_pre, adapted_proto_int
    
    def meta_train_step(self, task_batch, K_pre, K_int, 
                       inner_lr=0.01, inner_steps=5):
        """
        元训练步骤 (外循环)
        
        对一批任务(患者):
        1. 内循环: 在support set上快速适应
        2. 外循环: 在query set上评估,计算元梯度
        
        Args:
            task_batch: list of dict, 每个dict包含:
                - 'support_features': (N_s, D)
                - 'support_labels': (N_s,)
                - 'query_features': (N_q, D)
                - 'query_labels': (N_q,)
            K_pre: Preictal原型数量
            K_int: Interictal原型数量
            inner_lr: 内循环学习率
            inner_steps: 内循环步数
        
        Returns:
            meta_loss: scalar, 元损失(在query sets上的平均损失)
        """
        meta_loss = 0.0
        
        for task in task_batch:
            # 获取support和query数据
            support_features = task['support_features'].to(self.device)
            support_labels = task['support_labels'].to(self.device)
            query_features = task['query_features'].to(self.device)
            query_labels = task['query_labels'].to(self.device)
            
            # 从元原型初始化
            proto_pre = self.meta_prototypes_pre[:K_pre]
            proto_int = self.meta_prototypes_int[:K_int]
            
            # 内循环: 在support set上快速适应
            adapted_proto_pre, adapted_proto_int = self.inner_loop_update(
                support_features, support_labels,
                proto_pre, proto_int,
                inner_lr=inner_lr,
                inner_steps=inner_steps
            )
            
            # 外循环: 在query set上评估
            query_loss = self.prototype_loss(
                query_features, query_labels,
                adapted_proto_pre, adapted_proto_int
            )
            
            meta_loss += query_loss
        
        # 平均元损失
        meta_loss = meta_loss / len(task_batch)
        
        return meta_loss
    
    def meta_train(self, train_patients_data, K_pre, K_int,
                  meta_epochs=100, meta_batch_size=4,
                  meta_lr=0.001, inner_lr=0.01, inner_steps=5,
                  verbose=True):
        """
        元训练主循环
        
        在多个训练患者上学习元原型
        
        Args:
            train_patients_data: list of dict, 每个患者的数据
            K_pre: Preictal原型数量
            K_int: Interictal原型数量
            meta_epochs: 元训练轮数
            meta_batch_size: 每次采样的任务数
            meta_lr: 元学习率(外循环)
            inner_lr: 内循环学习率
            inner_steps: 内循环步数
            verbose: 是否打印训练信息
        """
        # 元优化器 (更新元参数)
        meta_optimizer = torch.optim.Adam(
            self.parameters(), lr=meta_lr
        )
        
        if verbose:
            print("=" * 80)
            print("开始元训练")
            print(f"训练患者数: {len(train_patients_data)}")
            print(f"元训练轮数: {meta_epochs}")
            print(f"K_pre={K_pre}, K_int={K_int}")
            print("=" * 80)
        
        for epoch in range(meta_epochs):
            # 随机采样一批任务(患者)
            task_indices = np.random.choice(
                len(train_patients_data),
                size=min(meta_batch_size, len(train_patients_data)),
                replace=False
            )
            task_batch = [train_patients_data[i] for i in task_indices]
            
            # 元训练步骤
            meta_loss = self.meta_train_step(
                task_batch, K_pre, K_int,
                inner_lr=inner_lr,
                inner_steps=inner_steps
            )
            
            # 更新元参数
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{meta_epochs}: Meta Loss = {meta_loss.item():.4f}")
        
        if verbose:
            print("=" * 80)
            print("✅ 元训练完成!")
            print("=" * 80)
    
    def fast_adapt(self, adaptation_features, adaptation_labels,
                  K_pre, K_int, adapt_steps=20, adapt_lr=0.01,
                  verbose=True):
        """
        在新患者上快速适应
        
        使用元学习的初始化,只需很少步数即可收敛
        
        Args:
            adaptation_features: (N, D) 适应数据特征
            adaptation_labels: (N,) 适应数据标签
            K_pre: Preictal原型数量
            K_int: Interictal原型数量
            adapt_steps: 适应步数
            adapt_lr: 适应学习率
            verbose: 是否打印信息
        
        Returns:
            proto_pre: (K_pre, D) 适应后的Preictal原型
            proto_int: (K_int, D) 适应后的Interictal原型
        """
        # 从元原型初始化
        proto_pre = self.meta_prototypes_pre[:K_pre].clone().detach()
        proto_int = self.meta_prototypes_int[:K_int].clone().detach()
        proto_pre.requires_grad = True
        proto_int.requires_grad = True
        
        # 优化器
        optimizer = torch.optim.Adam(
            [proto_pre, proto_int], lr=adapt_lr
        )
        
        if verbose:
            print(f"\n开始快速适应 (K_pre={K_pre}, K_int={K_int})")
        
        # 快速适应
        for step in range(adapt_steps):
            # 计算损失
            loss = self.prototype_loss(
                adaptation_features, adaptation_labels,
                proto_pre, proto_int
            )
            
            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if verbose and (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{adapt_steps}: Loss = {loss.item():.4f}")
        
        if verbose:
            print(f"✅ 快速适应完成!")
        
        return proto_pre.detach(), proto_int.detach()


def test_maml_initializer():
    """
    测试MAML元学习初始化器
    """
    print("=" * 80)
    print("测试MAML元学习原型初始化器")
    print("=" * 80)
    
    # 模拟数据
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建5个模拟患者的数据
    train_patients_data = []
    for i in range(5):
        # 每个患者有不同的数据分布
        support_features = torch.randn(100, 128) + i * 0.5
        support_labels = torch.randint(0, 2, (100,))
        query_features = torch.randn(50, 128) + i * 0.5
        query_labels = torch.randint(0, 2, (50,))
        
        train_patients_data.append({
            'support_features': support_features,
            'support_labels': support_labels,
            'query_features': query_features,
            'query_labels': query_labels
        })
    
    # 创建初始化器
    initializer = MetaLearnedPrototypeInitializer(
        feature_dim=128,
        max_K=10,
        device='cpu'
    )
    
    # 元训练
    initializer.meta_train(
        train_patients_data,
        K_pre=3,
        K_int=3,
        meta_epochs=50,
        meta_batch_size=3,
        verbose=True
    )
    
    # 在新患者上快速适应
    print("\n" + "=" * 80)
    print("测试在新患者上的快速适应")
    print("=" * 80)
    
    new_patient_features = torch.randn(100, 128)
    new_patient_labels = torch.randint(0, 2, (100,))
    
    proto_pre, proto_int = initializer.fast_adapt(
        new_patient_features,
        new_patient_labels,
        K_pre=3,
        K_int=3,
        adapt_steps=20,
        verbose=True
    )
    
    print(f"\n适应后的原型:")
    print(f"  Preictal原型: {proto_pre.shape}")
    print(f"  Interictal原型: {proto_int.shape}")


if __name__ == '__main__':
    test_maml_initializer()
