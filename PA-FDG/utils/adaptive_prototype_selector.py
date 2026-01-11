"""
自适应原型数量选择器 (基于MDL原则)

核心创新: 使用最小描述长度(Minimum Description Length)原则
自动确定每个患者的最优原型数量K

理论依据:
- Rissanen, J. (1978). "Modeling by shortest data description"
- MDL原则在统计学上等价于贝叶斯模型选择
- 渐近一致性: lim_{N→∞} P(K* = K_true) = 1

作者: AI Assistant
日期: 2024-11-19
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class AdaptivePrototypeSelector(nn.Module):
    """
    基于MDL原则的自适应原型数量选择器
    
    MDL(K) = Data_Cost(K) + Model_Cost(K)
    
    Data_Cost: 用K个原型描述数据的代价(重建误差)
    Model_Cost: 存储K个原型的代价(模型复杂度)
    
    最优K* = argmin_K MDL(K)
    """
    
    def __init__(self, max_K=10, min_K=1, random_state=42):
        """
        Args:
            max_K: 最大原型数量
            min_K: 最小原型数量
            random_state: 随机种子
        """
        super().__init__()
        self.max_K = max_K
        self.min_K = min_K
        self.random_state = random_state
    
    def compute_data_cost(self, features, prototypes):
        """
        计算数据代价(重建误差)
        
        Data_Cost = Σ_i min_k ||x_i - p_k||^2
        
        Args:
            features: (N, D) 特征向量
            prototypes: (K, D) 原型向量
        
        Returns:
            data_cost: float, 数据重建代价
        """
        # 计算每个样本到所有原型的距离
        distances = torch.cdist(features, prototypes)  # (N, K)
        
        # 每个样本到最近原型的距离
        min_distances = distances.min(dim=1)[0]  # (N,)
        
        # 数据代价 = 重建误差的平方和
        data_cost = min_distances.pow(2).sum()
        
        return data_cost.item()
    
    def compute_model_cost(self, K, D, N):
        """
        计算模型代价(编码K个原型的代价)
        
        Model_Cost = K * D * log(N)
        
        理论依据:
        - 编码K个D维原型需要K*D个参数
        - 每个参数需要log(N)比特来编码
        
        Args:
            K: 原型数量
            D: 特征维度
            N: 样本数量
        
        Returns:
            model_cost: float, 模型复杂度代价
        """
        model_cost = K * D * np.log(N)
        return model_cost
    
    def compute_mdl_score(self, features, K, prototypes):
        """
        计算MDL分数
        
        MDL(K) = Data_Cost(K) + Model_Cost(K)
        
        Args:
            features: (N, D) 特征向量
            K: 原型数量
            prototypes: (K, D) 原型向量
        
        Returns:
            mdl_score: float, MDL分数(越小越好)
        """
        N, D = features.shape
        
        # 数据代价
        data_cost = self.compute_data_cost(features, prototypes)
        
        # 模型代价
        model_cost = self.compute_model_cost(K, D, N)
        
        # MDL分数
        mdl_score = data_cost + model_cost
        
        return mdl_score
    
    def fit_kmeans(self, features, K):
        """
        使用K-means聚类得到K个原型
        
        Args:
            features: (N, D) 特征向量
            K: 原型数量
        
        Returns:
            prototypes: (K, D) 原型向量
        """
        # 转换为numpy
        features_np = features.detach().cpu().numpy()
        
        # K-means聚类
        kmeans = KMeans(
            n_clusters=K, 
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        kmeans.fit(features_np)
        
        # 获取聚类中心作为原型
        prototypes = torch.from_numpy(
            kmeans.cluster_centers_
        ).float().to(features.device)
        
        return prototypes
    
    def select_optimal_K(self, features, verbose=True):
        """
        自动选择最优K值
        
        算法:
        1. 对K ∈ [min_K, max_K]
        2. 使用K-means得到K个原型
        3. 计算MDL(K)
        4. 返回argmin_K MDL(K)
        
        Args:
            features: (N, D) 特征向量
            verbose: 是否打印详细信息
        
        Returns:
            best_K: int, 最优原型数量
            mdl_scores: list, 每个K对应的MDL分数
            all_prototypes: dict, 每个K对应的原型
        """
        N, D = features.shape
        
        if N < self.min_K:
            # 样本数太少，直接返回1
            if verbose:
                print(f"⚠️  样本数({N})小于min_K({self.min_K})，返回K=1")
            return 1, [0], {1: features.mean(dim=0, keepdim=True)}
        
        mdl_scores = []
        all_prototypes = {}
        
        # 遍历所有可能的K值
        K_range = range(self.min_K, min(self.max_K, N) + 1)
        
        for K in K_range:
            # K-means聚类
            prototypes = self.fit_kmeans(features, K)
            
            # 计算MDL分数
            mdl = self.compute_mdl_score(features, K, prototypes)
            mdl_scores.append(mdl)
            all_prototypes[K] = prototypes
            
            if verbose:
                data_cost = self.compute_data_cost(features, prototypes)
                model_cost = self.compute_model_cost(K, D, N)
                print(f"K={K}: MDL={mdl:.2f} (Data={data_cost:.2f}, Model={model_cost:.2f})")
        
        # 选择MDL最小的K
        best_idx = np.argmin(mdl_scores)
        best_K = list(K_range)[best_idx]
        
        if verbose:
            print(f"✅ 最优K值: {best_K} (MDL={mdl_scores[best_idx]:.2f})")
        
        return best_K, mdl_scores, all_prototypes
    
    def plot_mdl_curve(self, mdl_scores, best_K, save_path=None):
        """
        可视化MDL曲线
        
        Args:
            mdl_scores: list, MDL分数
            best_K: int, 最优K值
            save_path: str, 保存路径
        """
        K_range = range(self.min_K, self.min_K + len(mdl_scores))
        
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, mdl_scores, 'b-o', linewidth=2, markersize=8)
        plt.axvline(x=best_K, color='r', linestyle='--', linewidth=2, 
                   label=f'Optimal K={best_K}')
        plt.xlabel('Number of Prototypes (K)', fontsize=14)
        plt.ylabel('MDL Score', fontsize=14)
        plt.title('MDL-based Prototype Selection', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 MDL曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


def test_mdl_selector():
    """
    测试MDL选择器
    """
    print("=" * 80)
    print("测试MDL自适应原型选择器")
    print("=" * 80)
    
    # 生成模拟数据 (3个真实簇)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 簇1: 中心在(0, 0)
    cluster1 = torch.randn(100, 128) * 0.5
    
    # 簇2: 中心在(5, 5)
    cluster2 = torch.randn(80, 128) * 0.5 + 5
    
    # 簇3: 中心在(-5, 5)
    cluster3 = torch.randn(70, 128) * 0.5 + torch.tensor([-5, 5] + [0]*126)
    
    # 合并数据
    features = torch.cat([cluster1, cluster2, cluster3], dim=0)
    
    print(f"\n数据: {features.shape[0]}个样本, {features.shape[1]}维特征")
    print(f"真实簇数: 3\n")
    
    # 创建选择器
    selector = AdaptivePrototypeSelector(max_K=10, min_K=1)
    
    # 选择最优K
    best_K, mdl_scores, all_prototypes = selector.select_optimal_K(
        features, verbose=True
    )
    
    print(f"\n{'='*80}")
    print(f"结果: 最优K={best_K} (真实K=3)")
    print(f"{'='*80}")
    
    # 可视化MDL曲线
    selector.plot_mdl_curve(
        mdl_scores, best_K, 
        save_path='mdl_curve_test.png'
    )
    
    return best_K, mdl_scores


if __name__ == '__main__':
    test_mdl_selector()
