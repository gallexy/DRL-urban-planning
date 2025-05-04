import torch
import torch.nn as nn
import numpy as np
import math

from urban_planning.envs import city_config


class BaseStateEncoder(nn.Module):
    """
    基础状态编码器网络，作为所有编码器实现的基类
    """
    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(cfg)

        self.node_encoder = nn.Linear(agent.node_dim, cfg['gcn_node_dim'])
        self.max_num_nodes = cfg['max_num_nodes']
        self.max_num_edges = cfg['max_num_edges']

        self.output_policy_land_use_size = cfg['gcn_node_dim']*4
        self.output_policy_road_size = cfg['gcn_node_dim']
        self.output_value_size = cfg['gcn_node_dim']*2 + cfg['state_encoder_hidden_size'][-1] + 3

    def create_numerical_feature_encoder(self, cfg):
        """创建数值特征编码器"""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module(
                    'flatten_{}'.format(i),
                    nn.Flatten()
                )
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.agent.numerical_feature_size, cfg['state_encoder_hidden_size'][i])
                )
            else:
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['state_encoder_hidden_size'][i - 1], cfg['state_encoder_hidden_size'][i])
                )
            feature_encoder.add_module(
                'tanh_{}'.format(i),
                nn.Tanh()
            )
        return feature_encoder

    @staticmethod
    def batch_data(x):
        numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage = zip(*x)
        numerical_features = torch.stack(numerical_features)
        node_features = torch.stack(node_features)
        edge_index = torch.stack(edge_index)
        current_node_features = torch.stack(current_node_features)
        node_mask = torch.stack(node_mask)
        edge_mask = torch.stack(edge_mask)
        land_use_mask = torch.stack(land_use_mask)
        road_mask = torch.stack(road_mask)
        stage = torch.stack(stage)
        return numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage
            
    @staticmethod
    def mean_features(h, mask):
        mean_h = (h*mask.unsqueeze(-1).float()).sum(dim=1)/mask.float().sum(dim=1, keepdim=True)
        return mean_h


class MLPStateEncoder(BaseStateEncoder):
    """
    基于MLP的状态编码器网络
    """
    def __init__(self, cfg, agent):
        super().__init__(cfg, agent)

    def compute_edge_features(self, h_nodes, edge_index, edge_mask):
        """
        收集节点嵌入到边

        Args:
            h_nodes (torch.Tensor): 节点嵌入。形状: (batch, max_num_nodes, node_dim).
            edge_index (torch.Tensor): 边索引。形状: (batch, max_num_edges, 2).
            edge_mask (torch.Tensor): 边掩码。形状: (batch, max_num_edges).
            
        Returns:
            h_edges (torch.Tensor): 边嵌入。形状: (batch, max_num_edges, node_dim).
        """
        h_edges1 = torch.gather(h_nodes, 1, edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges2 = torch.gather(h_nodes, 1, edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        edges2_type = torch.eq(torch.argmax(h_edges2[:, :, :city_config.NUM_TYPES+1], dim=-1), city_config.FEASIBLE)
        edges2_type_mask = torch.broadcast_to(edges2_type.unsqueeze(-1), h_edges2.size())
        h_edges = torch.where(edges2_type_mask, h_edges2, h_edges1)
        mask = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape)
        h_edges = torch.where(mask, h_edges, torch.zeros_like(h_edges))
        return h_edges

    def forward(self, x):
        numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage = self.batch_data(x)
        h_numerical_features = self.numerical_feature_encoder(numerical_features)

        edge_features = self.compute_edge_features(node_features, edge_index, edge_mask)

        h_nodes = self.node_encoder(node_features)
        h_edges = self.node_encoder(edge_features)
        current_node_features = torch.unsqueeze(current_node_features, 1)
        h_current_node = self.node_encoder(current_node_features)

        h_edges_mean = self.mean_features(h_edges, edge_mask)
        h_nodes_mean = self.mean_features(h_nodes, node_mask)

        state_value = torch.cat([h_numerical_features, h_nodes_mean, h_edges_mean, stage],
                                dim=1)

        h_current_node_repeated = h_current_node.repeat(1, self.max_num_edges, 1)
        state_policy_land_use = torch.cat(
            [h_edges, h_current_node_repeated, h_edges*h_current_node_repeated, h_edges - h_current_node_repeated],
            dim=-1)

        state_policy_road = torch.cat([h_nodes], dim=-1)

        return state_policy_land_use, state_policy_road, state_value, land_use_mask, road_mask, stage


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为序列数据添加位置信息
    """
    def __init__(self, d_model, max_len=6000):  # 增大默认大小为10000
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.max_len = max_len
        
    def forward(self, x):
        # x形状: [batch_size, seq_len, feature_dim]
        if x.size(1) > self.max_len:
            # 如果输入序列太长，截断使用最大长度
            x = x[:, :self.max_len, :]
        return x + self.pe[:, :x.size(1), :]


class TransformerStateEncoder(BaseStateEncoder):
    """
    基于Transformer的状态编码器网络
    """
    def __init__(self, cfg, agent):
        super().__init__(cfg, agent)
        
        # 获取device信息（从上下文中传递）
        self.device = agent.device if hasattr(agent, 'device') else torch.device('cpu')
        
        # Transformer专用组件
        self.edge_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg['gcn_node_dim'], 
                nhead=cfg.get('transformer_heads', 2),
                dim_feedforward=cfg.get('transformer_dim_feedforward', 128),
                dropout=cfg.get('transformer_dropout', 0.1),
                batch_first=True
            ),
            num_layers=cfg.get('transformer_layers', 1)
        ).to(self.device)  # 显式移至GPU
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(cfg['gcn_node_dim']).to(self.device)  # 显式移至GPU
        
        # 设置输出尺寸，与基类保持一致
        self.output_policy_land_use_size = cfg['gcn_node_dim']  # 这里是变化的，因为我们直接输出transformer的结果
        self.output_policy_road_size = cfg['gcn_node_dim']
        self.output_value_size = cfg['gcn_node_dim']*3 + cfg['state_encoder_hidden_size'][-1] + 3
        
        # 打印设备信息以确认
        print(f"TransformerStateEncoder 初始化在设备: {self.device}")
        
    def compute_edge_features(self, h_nodes, edge_index, edge_mask):
        """
        收集节点嵌入到边
        
        Args:
            h_nodes (torch.Tensor): 节点嵌入。形状: (batch, max_num_nodes, node_dim).
            edge_index (torch.Tensor): 边索引。形状: (batch, max_num_edges, 2).
            edge_mask (torch.Tensor): 边掩码。形状: (batch, max_num_edges).
            
        Returns:
            h_edges (torch.Tensor): 边嵌入。形状: (batch, max_num_edges, node_dim).
        """
        h_edges1 = torch.gather(h_nodes, 1, edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges2 = torch.gather(h_nodes, 1, edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        edges2_type = torch.eq(torch.argmax(h_edges2[:, :, :city_config.NUM_TYPES+1], dim=-1), city_config.FEASIBLE)
        edges2_type_mask = torch.broadcast_to(edges2_type.unsqueeze(-1), h_edges2.size())
        h_edges = torch.where(edges2_type_mask, h_edges2, h_edges1)
        mask = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape)
        h_edges = torch.where(mask, h_edges, torch.zeros_like(h_edges))
        return h_edges
        
    def forward(self, x):
        numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage = self.batch_data(x)
            
        # 确保张量在正确的设备上
        device = stage.device  # 使用已经在GPU上的张量的设备
            
        # 基本特征提取
        h_numerical_features = self.numerical_feature_encoder(numerical_features)
        h_nodes = self.node_encoder(node_features)
        edge_features = self.compute_edge_features(node_features, edge_index, edge_mask)
        h_edges = self.node_encoder(edge_features)
        current_node_features = torch.unsqueeze(current_node_features, 1)
        h_current_node = self.node_encoder(current_node_features)
        
        # Transformer处理
        # 把当前对象特征作为特殊token和边特征拼接
        current_token = h_current_node.expand(-1, 1, -1)  # [batch, 1, feature_dim]
        sequence = torch.cat([current_token, h_edges], dim=1)  # [batch, edges+1, feature_dim]
        
        # 确保所有组件和张量都在同一设备上
        self.edge_transformer = self.edge_transformer.to(device)
        self.pos_encoder = self.pos_encoder.to(device)
        
        # 添加位置编码
        sequence = self.pos_encoder(sequence)
        
        # 创建注意力掩码（当前对象token不mask）
        attention_mask = torch.cat([
            torch.ones(edge_mask.shape[0], 1, device=device),
            edge_mask
        ], dim=1)
        attention_mask = ~attention_mask.bool()  # TransformerEncoder中True表示需要mask的位置
        
        # 使用原始的掩码方式，忽略嵌套张量警告
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 确保输入数据在GPU上
            sequence = sequence.to(device)
            attention_mask = attention_mask.to(device)
            transformer_output = self.edge_transformer(sequence, src_key_padding_mask=attention_mask)
        
        # 分离当前对象和边的表示
        h_current_transformed = transformer_output[:, 0:1, :]
        h_edges_transformed = transformer_output[:, 1:, :]
        
        # 计算全局特征
        h_edges_mean = self.mean_features(h_edges, edge_mask)
        h_nodes_mean = self.mean_features(h_nodes, node_mask)
        
        # 构建状态值特征
        state_value = torch.cat([h_numerical_features, h_nodes_mean, h_edges_mean, h_current_transformed.squeeze(1), stage],
                             dim=1)
        
        # 对于土地使用策略，直接使用transformer处理后的边特征
        state_policy_land_use = h_edges_transformed
        
        # 道路策略保持不变
        state_policy_road = torch.cat([h_nodes], dim=-1)
        
        return state_policy_land_use, state_policy_road, state_value, land_use_mask, road_mask, stage


class SGNNStateEncoder(nn.Module):
    """
    Single GNN state encoder network.
    """
    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(cfg)

        self.node_encoder = nn.Linear(agent.node_dim, cfg['gcn_node_dim'])
        self.num_gcn_layers = cfg['num_gcn_layers']
        self.num_edge_fc_layers = cfg['num_edge_fc_layers']
        self.edge_fc_layers = self.create_edge_fc_layers(cfg)
        self.max_num_nodes = cfg['max_num_nodes']
        self.max_num_edges = cfg['max_num_edges']

        self.attention_layer = nn.MultiheadAttention(cfg['gcn_node_dim'], cfg['num_attention_heads'])
        self.attention_query_layer = nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_key_layer = nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_value_layer = nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])

        self.output_policy_land_use_size = cfg['gcn_node_dim']*4
        self.output_policy_road_size = cfg['gcn_node_dim']
        self.output_value_size = cfg['gcn_node_dim']*3 + cfg['state_encoder_hidden_size'][-1] + 3

    def create_numerical_feature_encoder(self, cfg):
        """Create the numerical feature encoder."""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module(
                    'flatten_{}'.format(i),
                    nn.Flatten()
                )
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.agent.numerical_feature_size, cfg['state_encoder_hidden_size'][i])
                )
            else:
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['state_encoder_hidden_size'][i - 1], cfg['state_encoder_hidden_size'][i])
                )
            feature_encoder.add_module(
                'tanh_{}'.format(i),
                nn.Tanh()
            )
        return feature_encoder

    def create_edge_fc_layers(self, cfg):
        """Create the edge fc layers."""
        def create_edge_fc():
            seq = nn.Sequential()
            for i in range(self.num_edge_fc_layers):
                if i == 0:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim']*2, cfg['gcn_node_dim'])
                    )
                else:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
                    )
                seq.add_module(
                    'tanh_{}'.format(i),
                    nn.Tanh()
                )
            return seq
        edge_fc_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            edge_fc_layers.append(create_edge_fc())
        return edge_fc_layers

    def scatter_count(self, h_edges, indices, edge_mask, max_num_nodes):
        """
        Aggregate edge embeddings to nodes.

        Args:
            h_edges (torch.Tensor): Edge embeddings. Shape: (batch, max_num_edges, node_dim).
            indices (torch.Tensor): Node indices. Shape: (batch, max_num_edges).
            edge_mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            count_edge (torch.Tensor): Edge counts per node. Shape: (batch, max_num_nodes, node_dim).
        """
        batch_size = h_edges.shape[0]
        num_latents = h_edges.shape[2]

        h_nodes = torch.zeros(batch_size, max_num_nodes, num_latents).to(h_edges.device)
        count_edge = torch.zeros_like(h_nodes)
        count = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape).float()

        idx = indices.unsqueeze(-1).expand(-1, -1, num_latents)
        h_nodes = h_nodes.scatter_add_(1, idx, h_edges)
        count_edge = count_edge.scatter_add_(1, idx, count)
        return h_nodes, count_edge

    def gather_to_edges(self, h_nodes, edge_index, edge_mask, edge_fc_layer):
        """
        Gather node embeddings to edges.

        Args:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            edge_mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            edge_fc_layer (torch.nn.Module): Edge fc layer.

        Returns:
            h_edges (torch.Tensor): edge embeddings. Shape: (batch, max_num_edges, node_dim).
        """
        h_edges1 = torch.gather(h_nodes, 1, edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges2 = torch.gather(h_nodes, 1, edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_12 = torch.cat([h_edges1, h_edges2], -1)
        h_edges_21 = torch.cat([h_edges2, h_edges1], -1)
        h_edges = (edge_fc_layer(h_edges_12) + edge_fc_layer(h_edges_21)) / 2
        mask = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape)
        h_edges = torch.where(mask, h_edges, torch.zeros_like(h_edges))
        return h_edges

    def scatter_to_nodes(self, h_edges, edge_index, edge_mask, max_num_nodes):
        """
        Scatter edge embeddings to nodes.

        Args:
            h_edges (torch.Tensor): Edge embeddings. Shape: (batch, max_num_edges, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            edge_mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
        """
        h_nodes_1, count_1 = self.scatter_count(h_edges, edge_index[:, :, 0], edge_mask, max_num_nodes)
        h_nodes_2, count_2 = self.scatter_count(h_edges, edge_index[:, :, 1], edge_mask, max_num_nodes)
        h_nodes = (h_nodes_1 + h_nodes_2) / (count_1 + count_2 + self.EPSILON)
        return h_nodes

    def self_attention(self, h_current_node, h_nodes, node_mask):
        """Self attention."""
        query = self.attention_query_layer(h_current_node).transpose(0, 1)
        keys = self.attention_key_layer(h_nodes).transpose(0, 1)
        values = self.attention_value_layer(h_nodes).transpose(0, 1)
        h_current_node_attended, _ = self.attention_layer(
            query,
            keys,
            values,
            key_padding_mask=~node_mask)
        h_current_node_attended = h_current_node_attended.transpose(0, 1).squeeze(1)
        return h_current_node_attended

    @staticmethod
    def batch_data(x):
        numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage = zip(*x)
        numerical_features = torch.stack(numerical_features)
        node_features = torch.stack(node_features)
        edge_index = torch.stack(edge_index)
        current_node_features = torch.stack(current_node_features)
        node_mask = torch.stack(node_mask)
        edge_mask = torch.stack(edge_mask)
        land_use_mask = torch.stack(land_use_mask)
        road_mask = torch.stack(road_mask)
        stage = torch.stack(stage)
        return numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage

    @staticmethod
    def mean_features(h, mask):
        mean_h = (h*mask.unsqueeze(-1).float()).sum(dim=1)/mask.float().sum(dim=1, keepdim=True)
        return mean_h

    def forward(self, x):
        numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage = self.batch_data(x)
        h_numerical_features = self.numerical_feature_encoder(numerical_features)

        h_nodes = self.node_encoder(node_features)
        current_node_features = torch.unsqueeze(current_node_features, 1)
        h_current_node = self.node_encoder(current_node_features)

        # GCN
        for edge_fc_layer in self.edge_fc_layers:
            h_edges = self.gather_to_edges(h_nodes, edge_index, edge_mask, edge_fc_layer)
            h_nodes_new = self.scatter_to_nodes(h_edges, edge_index, edge_mask, self.max_num_nodes)
            h_nodes = h_nodes + h_nodes_new

        h_edges_mean = self.mean_features(h_edges, edge_mask)
        h_nodes_mean = self.mean_features(h_nodes, node_mask)

        h_current_node_attended = self.self_attention(h_current_node, h_nodes, node_mask)

        state_value = torch.cat([h_numerical_features, h_nodes_mean, h_edges_mean, h_current_node_attended, stage],
                                dim=1)

        h_current_node_repeated = h_current_node.repeat(1, self.max_num_edges, 1)
        state_policy_land_use = torch.cat(
            [h_edges, h_current_node_repeated, h_edges*h_current_node_repeated, h_edges - h_current_node_repeated],
            dim=-1)

        state_policy_road = torch.cat([h_nodes], dim=-1)

        return state_policy_land_use, state_policy_road, state_value, land_use_mask, road_mask, stage
