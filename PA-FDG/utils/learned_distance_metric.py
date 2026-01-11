"""
学习的距离度量 (Mahalanobis距离)

核心创新: 学习任务自适应的距离度量,而不是固定的欧氏距离

理论依据:
- Mahalanobis距离: d_M(x,p) = √((x-p)^T M (x-p))
- M是正定矩阵,等价于协方差矩阵的逆
- 自动学习特征的重要性和相关性结构

作者: AI Assistant  
日期: 2024-11-19
"""

import torch
import torch.nn as nn
import numpy as np


class LearnedMahalanobisDistance(nn.Module):
    """
    学习的Mahalanobis距离度量
    
    d_M(x, p) = √((x-p)^T M (x-p))
    
    其中M是学习的正定矩阵
    
    实现技巧:
    - 使用Cholesky分解: M = L @ L^T 保证正定性
    - L是下三角矩阵,可以直接学习
    """
    
    def __init__(self, feature_dim=128):
        """
        Args:
            feature_dim: 特征维度
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # 学习Cholesky分解的下三角矩阵L
        # M = L @ L^T 保证M正定
        self.L = nn.Parameter(torch.eye(feature_dim))
    
    def get_M(self):
        """
        获取正定矩阵M
        
        M = L @ L^T
        
        Returns:
            M: (D, D) 正定矩阵
        """
        M = self.L @ self.L.T
        return M
    
    def mahalanobis_distance(self, x, prototype):
        """
        计算Mahalanobis距离
        
        d_M(x, p) = √((x-p)^T M (x-p))
        
        Args:
            x: (N, D) 或 (D,) 查询向量
            prototype: (D,) 原型向量
        
        Returns:
            distances: (N,) 或 scalar
        """
        # 确保x是2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 计算差值
        diff = x - prototype.unsqueeze(0)  # (N, D)
        
        # 获取M矩阵
        M = self.get_M()  # (D, D)
        
        # 计算Mahalanobis距离
        # d^2 = (x-p)^T M (x-p)
        dist_squared = torch.sum(diff @ M * diff, dim=1)  # (N,)
        
        # 取平方根
        distances = torch.sqrt(torch.clamp(dist_squared, min=1e-8))
        
        if squeeze_output:
            distances = distances.squeeze(0)
        
        return distances
    
    def batch_mahalanobis_distance(self, x, prototypes):
        """
        批量计算到多个原型的Mahalanobis距离
        
        Args:
            x: (N, D) 查询向量
            prototypes: (K, D) 原型向量
        
        Returns:
            distances: (N, K) 距离矩阵
        """
        N = x.shape[0]
        K = prototypes.shape[0]
        
        distances = torch.zeros(N, K, device=x.device)
        
        for k in range(K):
            distances[:, k] = self.mahalanobis_distance(x, prototypes[k])
        
        return distances
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        从M矩阵的对角线元素可以看出特征的重要性
        
        Returns:
            importance: (D,) 特征重要性分数
        """
        M = self.get_M()
        importance = torch.diag(M)
        return importance
    
    def visualize_M_matrix(self, save_path=None):
        """
        可视化M矩阵
        
        Args:
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        M = self.get_M().detach().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(M, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='Matrix Value')
        plt.title('Learned Mahalanobis Distance Matrix M', fontsize=16)
        plt.xlabel('Feature Dimension', fontsize=14)
        plt.ylabel('Feature Dimension', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 M矩阵已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


class DiagonalMahalanobisDistance(nn.Module):
    """
    对角Mahalanobis距离 (简化版本)
    
    假设特征之间独立,M是对角矩阵
    
    d_M(x, p) = √(Σ_i w_i (x_i - p_i)^2)
    
    其中w_i是学习的特征权重
    
    优势:
    - 参数更少 (D vs D^2)
    - 计算更快
    - 可解释性更强 (直接看出特征重要性)
    """
    
    def __init__(self, feature_dim=128):
        """
        Args:
            feature_dim: 特征维度
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # 学习特征权重 (对数空间,保证正性)
        self.log_weights = nn.Parameter(torch.zeros(feature_dim))
    
    def get_weights(self):
        """
        获取特征权重
        
        Returns:
            weights: (D,) 正的特征权重
        """
        weights = torch.exp(self.log_weights)
        return weights
    
    def diagonal_mahalanobis_distance(self, x, prototype):
        """
        计算对角Mahalanobis距离
        
        d(x, p) = √(Σ_i w_i (x_i - p_i)^2)
        
        Args:
            x: (N, D) 或 (D,) 查询向量
            prototype: (D,) 原型向量
        
        Returns:
            distances: (N,) 或 scalar
        """
        # 确保x是2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 计算差值
        diff = x - prototype.unsqueeze(0)  # (N, D)
        
        # 获取权重
        weights = self.get_weights()  # (D,)
        
        # 加权平方和
        weighted_squared_diff = weights * diff.pow(2)  # (N, D)
        dist_squared = weighted_squared_diff.sum(dim=1)  # (N,)
        
        # 取平方根
        distances = torch.sqrt(torch.clamp(dist_squared, min=1e-8))
        
        if squeeze_output:
            distances = distances.squeeze(0)
        
        return distances
    
    def batch_diagonal_mahalanobis_distance(self, x, prototypes):
        """
        批量计算对角Mahalanobis距离
        
        Args:
            x: (N, D)
            prototypes: (K, D)
        
        Returns:
            distances: (N, K)
        """
        N = x.shape[0]
        K = prototypes.shape[0]
        
        distances = torch.zeros(N, K, device=x.device)
        
        for k in range(K):
            distances[:, k] = self.diagonal_mahalanobis_distance(
                x, prototypes[k]
            )
        
        return distances
    
    def get_feature_importance(self):
        """
        获取特征重要性 (就是权重本身)
        
        Returns:
            importance: (D,) 特征重要性
        """
        return self.get_weights()
    
    def visualize_feature_importance(self, save_path=None, top_k=20):
        """
        可视化特征重要性
        
        Args:
            save_path: 保存路径
            top_k: 显示前k个最重要的特征
        """
        import matplotlib.pyplot as plt
        
        importance = self.get_feature_importance().detach().cpu().numpy()
        
        # 排序
        sorted_indices = np.argsort(importance)[::-1]
        top_indices = sorted_indices[:top_k]
        top_importance = importance[top_indices]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(top_k), top_importance)
        plt.xlabel('Feature Index', fontsize=14)
        plt.ylabel('Importance Weight', fontsize=14)
        plt.title(f'Top {top_k} Feature Importance', fontsize=16)
        plt.xticks(range(top_k), top_indices, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 特征重要性已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


def test_distance_metrics():
    """
    测试距离度量
    """
    print("=" * 80)
    print("测试学习的距离度量")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    # 创建测试数据
    feature_dim = 128
    x = torch.randn(100, feature_dim)
    prototypes = torch.randn(3, feature_dim)
    
    # 测试完整Mahalanobis距离
    print("\n1. 完整Mahalanobis距离")
    print("-" * 80)
    mahal_metric = LearnedMahalanobisDistance(feature_dim)
    
    distances = mahal_metric.batch_mahalanobis_distance(x, prototypes)
    print(f"距离矩阵: {distances.shape}")
    print(f"距离范围: [{distances.min():.4f}, {distances.max():.4f}]")
    
    # 可视化M矩阵
    mahal_metric.visualize_M_matrix('mahalanobis_M_matrix.png')
    
    # 测试对角Mahalanobis距离
    print("\n2. 对角Mahalanobis距离")
    print("-" * 80)
    diag_metric = DiagonalMahalanobisDistance(feature_dim)
    
    distances_diag = diag_metric.batch_diagonal_mahalanobis_distance(
        x, prototypes
    )
    print(f"距离矩阵: {distances_diag.shape}")
    print(f"距离范围: [{distances_diag.min():.4f}, {distances_diag.max():.4f}]")
    
    # 可视化特征重要性
    diag_metric.visualize_feature_importance(
        'feature_importance.png', top_k=20
    )
    
    # 对比欧氏距离
    print("\n3. 对比欧氏距离")
    print("-" * 80)
    euclidean_distances = torch.cdist(x, prototypes)
    print(f"欧氏距离范围: [{euclidean_distances.min():.4f}, {euclidean_distances.max():.4f}]")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成!")
    print("=" * 80)


if __name__ == '__main__':
    test_distance_metrics()
