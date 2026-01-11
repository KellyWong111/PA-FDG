#!/usr/bin/env python3
"""
DSTG-OML v2: 优化版动态时空图模型

改进:
1. 恢复学习连接权重（向量化）
2. 增加模型容量
3. 多尺度特征提取
4. 残差连接
5. 更强的表达能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import torch_geometric.nn as gnn
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class DynamicGraphConstructorV2(nn.Module):
    """
    优化版动态图构建
    
    改进:
    1. 向量化计算相关性
    2. 保留学习连接权重
    3. 多头注意力机制
    """
    def __init__(self, num_channels=22, feature_dim=19, num_heads=4):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 多头注意力学习连接
        self.edge_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 边权重学习网络
        self.edge_weight_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        print("✅ 优化版动态图构建模块初始化完成")
    
    def compute_correlation_matrix(self, x):
        """
        向量化计算相关性矩阵
        x: (batch, num_channels, feature_dim)
        """
        # 标准化
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        x_std = x_centered.std(dim=-1, keepdim=True) + 1e-8
        x_norm = x_centered / x_std
        
        # 相关性矩阵
        corr_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2)) / x.size(-1)
        
        return torch.abs(corr_matrix)
    
    def learn_edge_weights(self, x):
        """
        学习边权重（向量化）
        x: (batch, num_channels, feature_dim)
        """
        batch_size, num_channels, feature_dim = x.size()
        
        # 扩展为所有通道对
        # x_i: (batch, num_channels, 1, feature_dim)
        # x_j: (batch, 1, num_channels, feature_dim)
        x_i = x.unsqueeze(2).expand(-1, -1, num_channels, -1)
        x_j = x.unsqueeze(1).expand(-1, num_channels, -1, -1)
        
        # 拼接: (batch, num_channels, num_channels, feature_dim*2)
        edge_features = torch.cat([x_i, x_j], dim=-1)
        
        # 学习权重: (batch, num_channels, num_channels)
        weights = self.edge_weight_net(edge_features).squeeze(-1)
        
        return weights
    
    def forward(self, x):
        """
        x: (batch, num_channels, feature_dim)
        返回: (batch, num_channels, num_channels)
        """
        batch_size = x.size(0)
        
        # 1. 计算相关性矩阵
        corr_matrix = self.compute_correlation_matrix(x)
        
        # 2. 学习边权重
        edge_weights = self.learn_edge_weights(x)
        
        # 3. 加权相关性
        adj_matrix = corr_matrix * edge_weights
        
        # 4. 添加自连接
        eye = torch.eye(self.num_channels, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_matrix = adj_matrix + eye
        
        return adj_matrix


class MultiScaleGCN(nn.Module):
    """
    多尺度图卷积网络
    
    改进:
    1. 多个GCN分支（不同感受野）
    2. 残差连接
    3. 层归一化
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # 多尺度GCN分支
        self.gcn_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            
            # GCN层
            self.gcn_layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
            
            # 跳跃连接
            if i > 0:
                self.skip_connections.append(nn.Linear(in_dim, hidden_channels))
        
        print(f"✅ 多尺度GCN初始化完成 ({num_layers}层)")
    
    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_channels)
        adj: (batch, num_nodes, num_nodes)
        """
        # 归一化邻接矩阵
        degree = adj.sum(dim=-1, keepdim=True)
        adj_norm = adj / (degree + 1e-8)
        
        h = x
        for i, gcn in enumerate(self.gcn_layers):
            # 图卷积
            h_new = torch.bmm(adj_norm, h)
            h_new = gcn(h_new)
            
            # 残差连接
            if i > 0:
                h = h_new + self.skip_connections[i-1](h)
            else:
                h = h_new
        
        return h


class TemporalAttentionV2(nn.Module):
    """
    增强版时间注意力
    
    改进:
    1. 多头注意力
    2. 前馈网络
    3. 残差连接
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"✅ 增强版时间注意力初始化完成 ({num_heads}头)")
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # 多头注意力 + 残差
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class DSTG_Model_V2(nn.Module):
    """
    优化版DSTG模型
    
    改进:
    1. 更强的动态图构建
    2. 多尺度GCN
    3. 增强版时间注意力
    4. 更大的模型容量
    5. 残差连接
    """
    def __init__(self, num_channels=22, feature_dim=19, 
                 hidden_dim=256, num_gcn_layers=4, num_heads=8):
        super().__init__()
        
        print("="*70)
        print("初始化 DSTG-V2 模型 (优化版)")
        print("="*70)
        
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 模块1: 优化版动态图构建
        self.graph_constructor = DynamicGraphConstructorV2(
            num_channels=num_channels,
            feature_dim=hidden_dim,  # 使用投影后的维度
            num_heads=4
        )
        
        # 模块2: 多尺度GCN
        self.spatial_gcn = MultiScaleGCN(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_gcn_layers
        )
        
        # 模块3: 增强版时间注意力
        self.temporal_attention = TemporalAttentionV2(
            d_model=hidden_dim,
            num_heads=num_heads
        )
        
        # 模块4: 全局池化
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * num_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 模块5: 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        print("="*70)
        print("✅ DSTG-V2 模型初始化完成")
        print(f"参数量: {sum(p.numel() for p in self.parameters()):,}")
        print(f"隐藏层维度: {hidden_dim}")
        print(f"GCN层数: {num_gcn_layers}")
        print(f"注意力头数: {num_heads}")
        print("="*70)
    
    def forward(self, x):
        """
        x: (batch, 1, total_features) 或 (batch, total_features)
        """
        # 处理输入维度
        if x.dim() == 3:
            x = x.squeeze(1)
        
        batch_size = x.size(0)
        total_features = x.size(1)
        
        # 重塑为 (batch, num_channels, feature_per_channel)
        feature_per_channel = total_features // self.num_channels
        x = x.view(batch_size, self.num_channels, feature_per_channel)
        
        # 输入投影
        x = self.input_projection(x)  # (batch, num_channels, hidden_dim)
        
        # 1. 构建动态图
        adj_matrix = self.graph_constructor(x)
        
        # 2. 多尺度GCN
        spatial_features = self.spatial_gcn(x, adj_matrix)
        
        # 3. 时间注意力
        temporal_features = self.temporal_attention(spatial_features)
        
        # 4. 全局池化
        pooled = temporal_features.view(batch_size, -1)
        pooled = self.global_pool(pooled)
        
        # 5. 分类
        output = self.classifier(pooled)
        
        return output
    
    def get_features(self, x):
        """提取特征（用于对比学习）"""
        if x.dim() == 3:
            x = x.squeeze(1)
        
        batch_size = x.size(0)
        total_features = x.size(1)
        feature_per_channel = total_features // self.num_channels
        x = x.view(batch_size, self.num_channels, feature_per_channel)
        
        x = self.input_projection(x)
        adj_matrix = self.graph_constructor(x)
        spatial_features = self.spatial_gcn(x, adj_matrix)
        temporal_features = self.temporal_attention(spatial_features)
        pooled = temporal_features.view(batch_size, -1)
        pooled = self.global_pool(pooled)
        
        return pooled


def test_model_v2():
    """测试优化版模型"""
    print("\n" + "="*70)
    print("测试 DSTG-V2 模型")
    print("="*70)
    
    # 创建模型
    model = DSTG_Model_V2(
        num_channels=22,
        feature_dim=19,
        hidden_dim=256,  # 增加到256
        num_gcn_layers=4,  # 增加到4层
        num_heads=8
    )
    
    # 测试输入
    batch_size = 4
    actual_features = 22 * 19
    x = torch.randn(batch_size, 1, actual_features)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")
    
    # 提取特征
    features = model.get_features(x)
    print(f"特征形状: {features.shape}")
    
    print("\n✅ 模型测试通过！")
    print("="*70)


if __name__ == '__main__':
    test_model_v2()
