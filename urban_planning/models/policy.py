import torch
import torch.nn as nn
import math


class TransformerPolicyHead(nn.Module):
    """基于Transformer的策略头"""
    def __init__(self, input_size, hidden_size, nhead=2, num_layers=1):
        super().__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward_old(self, x, mask=None):
        # 投影到Transformer维度
        print("Transformer input shape:", x.shape, mask)
        x = self.input_proj(x)
        
        # 创建注意力掩码
        if mask is not None:
            attention_mask = ~mask.bool()  # Transformer中True表示需要mask的位置
        else:
            attention_mask = None
            
        # 应用Transformer
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # 应用输出层
        logits = self.output_layer(x).squeeze(-1)  # [batch, seq_len]
        
        return logits
    def forward(self, x, mask=None):
        # x: [batch, seq_len, input_dim]
        # mask: [batch, seq_len]  True表示有效

        batch_size, seq_len, _ = x.shape
        device = x.device

        #print("Transformer input shape:", x.shape, mask)

        x_proj = self.input_proj(x)

        # 初始化输出
        logits = torch.zeros(batch_size, seq_len, device=device, dtype=x_proj.dtype)

        if mask is None:
            # 没有mask，直接全量送入Transformer
            x_trans = self.transformer(x_proj)
            logits = self.output_layer(x_trans).squeeze(-1)
            return logits
        
        #print("True in mask:", mask.sum().item(), mask.shape)
        #print("False in mask:", (~mask).sum().item())


        # 动态裁剪：只对有效token做注意力
        for i in range(batch_size):
            valid_idx = mask[i].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue  # 本样本全无效
            x_valid = x_proj[i, valid_idx].unsqueeze(0)  # [1, num_valid, dim]
            # TransformerEncoder 需要3D输入
            x_valid_trans = self.transformer(x_valid)
            logit_valid = self.output_layer(x_valid_trans).squeeze(-1)  # [1, num_valid] → [num_valid]
            logits[i, valid_idx] = logit_valid

        return logits


class UrbanPlanningPolicy(nn.Module):
    """
    Policy network for urban planning.
    """
    def __init__(self, cfg, agent, shared_net):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.shared_net = shared_net
        
        # 使用Transformer策略头
        self.use_transformer = cfg.get('use_transformer_policy', False)
        
        if self.use_transformer:
            self.policy_land_use_head = TransformerPolicyHead(
                self.shared_net.output_policy_land_use_size, 
                cfg.get('transformer_policy_hidden_size', 4),
                nhead=cfg.get('transformer_policy_heads', 2),
                num_layers=cfg.get('transformer_policy_layers', 1)
            )
            
            self.policy_road_head = TransformerPolicyHead(
                self.shared_net.output_policy_road_size,
                cfg.get('transformer_policy_hidden_size', 4),
                nhead=cfg.get('transformer_policy_heads', 2),
                num_layers=cfg.get('transformer_policy_layers', 1)
            )
        else:
            # 保留原始MLP策略头作为备选
            self.policy_land_use_head = self.create_policy_head(
                self.shared_net.output_policy_land_use_size, cfg['policy_land_use_head_hidden_size'], 'land_use')
            self.policy_road_head = self.create_policy_head(
                self.shared_net.output_policy_road_size, cfg['policy_road_head_hidden_size'], 'road')

    def create_policy_head(self, input_size, hidden_size, name):
        """Create the policy land_use head."""
        policy_head = nn.Sequential()
        for i in range(len(hidden_size)):
            if i == 0:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(input_size, hidden_size[i])
                )
            else:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(hidden_size[i - 1], hidden_size[i], bias=False)
                )
            if i < len(hidden_size) - 1:
                policy_head.add_module(
                    '{}_tanh_{}'.format(name, i),
                    nn.Tanh()
                )
            elif hidden_size[i] == 1:
                policy_head.add_module(
                    '{}_flatten_{}'.format(name, i),
                    nn.Flatten()
                )
        return policy_head

    def forward(self, x):
        state_policy_land_use, state_policy_road, _, land_use_mask, road_mask, stage = self.shared_net(x)

        if stage[:, 0].sum() > 0:
            if self.use_transformer:
                # 使用Transformer策略头处理
                land_use_logits = self.policy_land_use_head(
                    state_policy_land_use[stage[:, 0].bool()], 
                    land_use_mask[stage[:, 0].bool()]
                )
            else:
                # 使用原始MLP策略头处理
                land_use_logits = self.policy_land_use_head(state_policy_land_use[stage[:, 0].bool()])
                
            land_use_paddings = torch.ones_like(land_use_mask[stage[:, 0].bool()], dtype=self.agent.dtype)*(-2.**32+1)
            masked_land_use_logits = torch.where(land_use_mask[stage[:, 0].bool()], land_use_logits, land_use_paddings)
            land_use_dist = torch.distributions.Categorical(logits=masked_land_use_logits)
        else:
            land_use_dist = None

        if stage[:, 1].sum() > 0:
            if self.use_transformer:
                # 使用Transformer策略头处理
                road_logits = self.policy_road_head(
                    state_policy_road[stage[:, 1].bool()],
                    road_mask[stage[:, 1].bool()]
                )
            else:
                # 使用原始MLP策略头处理
                road_logits = self.policy_road_head(state_policy_road[stage[:, 1].bool()])
                
            road_paddings = torch.ones_like(road_mask[stage[:, 1].bool()], dtype=self.agent.dtype)*(-2.**32 + 1)
            masked_road_logits = torch.where(road_mask[stage[:, 1].bool()], road_logits, road_paddings)
            road_dist = torch.distributions.Categorical(logits=masked_road_logits)
        else:
            road_dist = None

        return land_use_dist, road_dist, stage

    def select_action(self, x, mean_action=False):
        land_use_dist, road_dist, stage = self.forward(x)
        batch_size = stage.shape[0]
        action = torch.zeros(batch_size, 2, dtype=self.agent.dtype, device=stage.device)
        if land_use_dist is not None:
            if mean_action:
                land_use_action = land_use_dist.probs.argmax(dim=1).to(self.agent.dtype)
            else:
                land_use_action = land_use_dist.sample().to(self.agent.dtype)
            action[stage[:, 0].bool(), 0] = land_use_action

        if road_dist is not None:
            if mean_action:
                road_action = road_dist.probs.argmax(dim=1).to(self.agent.dtype)
            else:
                road_action = road_dist.sample().to(self.agent.dtype)
            action[stage[:, 1].bool(), 1] = road_action

        return action

    def get_log_prob_entropy(self, x, action):
        land_use_dist, road_dist, stage = self.forward(x)
        batch_size = stage.shape[0]
        log_prob = torch.zeros(batch_size, dtype=self.agent.dtype, device=stage.device)
        entropy = torch.zeros(batch_size, dtype=self.agent.dtype, device=stage.device)
        if land_use_dist is not None:
            land_use_action = action[stage[:, 0].bool(), 0]
            land_use_log_prob = land_use_dist.log_prob(land_use_action)
            log_prob[stage[:, 0].bool()] = land_use_log_prob
            entropy[stage[:, 0].bool()] = land_use_dist.entropy()

        if road_dist is not None:
            road_action = action[stage[:, 1].bool(), 1]
            road_log_prob = road_dist.log_prob(road_action)
            log_prob[stage[:, 1].bool()] = road_log_prob
            entropy[stage[:, 1].bool()] = road_dist.entropy()

        return log_prob.unsqueeze(1), entropy.unsqueeze(1)
