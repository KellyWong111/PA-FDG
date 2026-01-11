"""
DSTG-V2 Hybrid Graph Model
混合图架构：融合相关性图 + 语义图

核心创新:
1. Correlation Graph: 基于统计相关性（V13）
2. Semantic Graph: 基于通道嵌入 + 注意力（V24）
3. Adaptive Fusion: 自适应融合两种图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CorrelationGraphConstructor(nn.Module):
    """
    相关性图构建器（V13 风格）
    基于统计相关性 + 学习边权重
    """
    def __init__(self, num_channels=22, feature_dim=448, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        
        # 边权重学习网络
        self.edge_weight_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        print("✅ 相关性图构建器初始化完成")
    
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
        Returns: (batch, num_channels, num_channels)
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


class SemanticGraphConstructor(nn.Module):
    """
    语义图构建器（V24 风格）
    基于通道嵌入 + 注意力机制
    """
    def __init__(self, num_channels=22, feature_dim=448, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        
        # 每个通道的可学习嵌入（关键创新！）
        self.channel_embedding = nn.Parameter(
            torch.randn(num_channels, hidden_dim) * 0.01
        )
        
        # 边权重计算网络
        edge_input_dim = hidden_dim * 2 + feature_dim * 2
        
        self.edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 稀疏化阈值（可学习）
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.3))
        
        print(f"✅ 语义图构建器初始化完成 (嵌入维度: {hidden_dim})")
    
    def forward(self, x):
        """
        x: (batch, num_channels, feature_dim)
        Returns: (batch, num_channels, num_channels)
        """
        batch_size, num_channels, feature_dim = x.size()
        
        # 扩展通道嵌入到 batch
        channel_emb = self.channel_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 构建邻接矩阵
        adj = torch.zeros(batch_size, num_channels, num_channels).to(x.device)
        
        # 计算所有边的权重
        for i in range(num_channels):
            for j in range(i+1, num_channels):
                # 拼接: 嵌入i + 嵌入j + 特征i + 特征j
                edge_input = torch.cat([
                    channel_emb[:, i],
                    channel_emb[:, j],
                    x[:, i],
                    x[:, j]
                ], dim=1)
                
                # 计算边权重
                weight = self.edge_network(edge_input).squeeze(1)
                
                adj[:, i, j] = weight
                adj[:, j, i] = weight  # 对称
        
        # 稀疏化
        threshold = torch.sigmoid(self.sparsity_threshold)
        adj = torch.where(
            adj > threshold,
            adj,
            torch.zeros_like(adj)
        )
        
        # 添加自连接
        eye = torch.eye(num_channels).unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        adj = adj + eye
        
        return adj


class HybridGraphConstructor(nn.Module):
    """
    混合图构建器（核心创新！）
    自适应融合相关性图 + 语义图
    
    创新点:
    1. 融合统计相关性 + 学习语义关系
    2. 自适应学习融合权重
    3. 充分利用两种图的优势
    """
    def __init__(self, num_channels=22, feature_dim=448, hidden_dim=64, 
                 fusion_mode='learned', fixed_alpha=0.6, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.fusion_mode = fusion_mode
        
        print("="*70)
        print("🔥 初始化混合图构建器 (Hybrid Graph Constructor)")
        print("="*70)
        
        # 相关性图分支
        self.corr_graph = CorrelationGraphConstructor(
            num_channels=num_channels,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        # 语义图分支
        self.semantic_graph = SemanticGraphConstructor(
            num_channels=num_channels,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout * 3  # 语义图用更高的 dropout
        )
        
        # 融合策略
        if fusion_mode == 'fixed':
            # 固定权重融合
            self.alpha = fixed_alpha
            print(f"✅ 融合模式: 固定权重 (α={fixed_alpha:.2f})")
        elif fusion_mode == 'learned':
            # 学习融合权重（全局）
            self.alpha_param = nn.Parameter(torch.tensor(0.6))
            print("✅ 融合模式: 学习全局权重")
        elif fusion_mode == 'adaptive':
            # 自适应融合（基于特征）
            self.fusion_net = nn.Sequential(
                nn.Linear(feature_dim * num_channels, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            print("✅ 融合模式: 自适应融合网络")
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        
        self.fusion_mode = fusion_mode
        
        print("="*70)
        print("✅ 混合图构建器初始化完成！")
        print(f"   - 相关性图: 统计相关性 + 学习边权重")
        print(f"   - 语义图: 通道嵌入 + 注意力机制")
        print(f"   - 融合策略: {fusion_mode}")
        print("="*70)
    
    def compute_fusion_weight(self, x):
        """
        计算融合权重 α
        α = 1: 完全使用相关性图
        α = 0: 完全使用语义图
        """
        if self.fusion_mode == 'fixed':
            return self.alpha
        elif self.fusion_mode == 'learned':
            return torch.sigmoid(self.alpha_param)
        elif self.fusion_mode == 'adaptive':
            # 基于特征自适应计算权重
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)  # (batch, num_channels * feature_dim)
            alpha = self.fusion_net(x_flat).squeeze(1)  # (batch,)
            return alpha
        else:
            return 0.6
    
    def forward(self, x):
        """
        x: (batch, num_channels, feature_dim)
        Returns: (batch, num_channels, num_channels)
        """
        # 1. 构建相关性图
        G_corr = self.corr_graph(x)
        
        # 2. 构建语义图
        G_sem = self.semantic_graph(x)
        
        # 3. 计算融合权重
        alpha = self.compute_fusion_weight(x)
        
        # 4. 融合两种图
        if self.fusion_mode == 'adaptive':
            # 自适应融合（每个样本不同权重）
            alpha = alpha.view(-1, 1, 1)  # (batch, 1, 1)
            G_hybrid = alpha * G_corr + (1 - alpha) * G_sem
        else:
            # 固定或学习的全局权重
            G_hybrid = alpha * G_corr + (1 - alpha) * G_sem
        
        return G_hybrid


class MultiScaleGCN(nn.Module):
    """多尺度图卷积网络"""
    def __init__(self, in_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            self.gcn_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
    
    def forward(self, x, adj):
        """
        x: (batch, num_channels, feature_dim)
        adj: (batch, num_channels, num_channels)
        """
        for i, gcn_layer in enumerate(self.gcn_layers):
            # 图卷积: x' = adj @ x
            x_agg = torch.bmm(adj, x)
            # 特征变换
            x = gcn_layer(x_agg)
        
        return x


class EnhancedTemporalAttention(nn.Module):
    """增强版时间注意力"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch, num_channels, hidden_dim)
        """
        # 多头注意力
        attn_out, _ = self.attention(x, x, x)
        
        # 残差连接 + 层归一化
        x = self.norm(x + self.dropout(attn_out))
        
        return x


class DSTGV2Hybrid(nn.Module):
    """
    DSTG-V2 混合图模型
    
    核心创新:
    1. 混合图构建（相关性 + 语义）
    2. 多尺度 GCN
    3. 增强时间注意力
    """
    def __init__(self, num_channels=22, feature_dim=448, hidden_dim=128, 
                 output_dim=128, fusion_mode='learned', fixed_alpha=0.6):
        super().__init__()
        
        print("\n" + "="*70)
        print("🚀 初始化 DSTG-V2 混合图模型")
        print("="*70)
        
        # 混合图构建器（核心创新！）
        self.graph_constructor = HybridGraphConstructor(
            num_channels=num_channels,
            feature_dim=feature_dim,
            hidden_dim=64,
            fusion_mode=fusion_mode,
            fixed_alpha=fixed_alpha
        )
        
        # 多尺度 GCN
        self.gcn = MultiScaleGCN(
            in_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=3
        )
        print("✅ 多尺度GCN初始化完成 (3层)")
        
        # 增强版时间注意力
        self.temporal_attention = EnhancedTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=8
        )
        print("✅ 增强版时间注意力初始化完成 (8头)")
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        
        print("="*70)
        print("✅ DSTG-V2 混合图模型初始化完成")
        print(f"   参数量: {total_params:,}")
        print(f"   隐藏层维度: {hidden_dim}")
        print(f"   GCN层数: 3")
        print(f"   注意力头数: 8")
        print(f"   融合模式: {fusion_mode}")
        print("="*70 + "\n")
    
    def forward(self, x):
        """
        x: (batch, num_channels, feature_dim)
        Returns: (batch, output_dim)
        """
        batch_size, num_channels, feature_dim = x.size()
        
        # 1. 构建混合图
        adj_matrix = self.graph_constructor(x)
        
        # 2. 多尺度 GCN
        x_gcn = self.gcn(x, adj_matrix)
        
        # 3. 时间注意力
        x_attn = self.temporal_attention(x_gcn)
        
        # 4. 全局池化 + 输出投影
        x_flat = x_attn.view(batch_size, -1)
        output = self.output_proj(x_flat)
        
        return output
    
    def get_features(self, x):
        """
        提取特征（用于对比学习和原型计算）
        与 forward 相同
        """
        return self.forward(x)
