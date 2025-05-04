import torch
import torch.nn as nn
from urban_planning.models.state_encoder import SGNNStateEncoder, MLPStateEncoder, TransformerStateEncoder
from urban_planning.models.policy import UrbanPlanningPolicy
from urban_planning.models.value import UrbanPlanningValue


def create_sgnn_model(cfg, agent):
    """从配置文件创建策略和价值网络
    Args:
        cfg: 配置对象
        agent: 智能体对象
    Returns:
        包含策略网络和价值网络的元组
    """
    # 为了保持兼容性，确保使用SGNN编码器
    cfg.state_encoder_specs['encoder_type'] = 'transformer'
    shared_net = create_state_encoder(cfg, agent)
    policy_net = UrbanPlanningPolicy(cfg.policy_specs, agent, shared_net)
    value_net = UrbanPlanningValue(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


def create_state_encoder(cfg, agent):
    """工厂函数，根据配置创建对应的状态编码器
    Args:
        cfg: 配置对象
        agent: 智能体对象
    Returns:
        适合的状态编码器实例
    """
    encoder_type = cfg.state_encoder_specs.get('encoder_type', 'transformer')
    #encoder_type = 'transformer'
    if encoder_type == 'sgnn':
        return SGNNStateEncoder(cfg.state_encoder_specs, agent)
    elif encoder_type == 'transformer':
        return TransformerStateEncoder(cfg.state_encoder_specs, agent)
    else:  # 默认使用MLP编码器
        return MLPStateEncoder(cfg.state_encoder_specs, agent)


def create_mlp_model(cfg, agent):
    """创建一个基于MLP的模型
    Args:
        cfg: 配置对象
        agent: 智能体对象
    Returns:
        包含策略网络和价值网络的元组
    """
    shared_net = create_state_encoder(cfg, agent)
    policy_net = UrbanPlanningPolicy(cfg.policy_specs, agent, shared_net)
    value_net = UrbanPlanningValue(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


class ActorCritic(nn.Module):
    """
    An Actor-Critic network for parsing parameters.

    Args:
        actor_net (nn.Module): actor network.
        value_net (nn.Module): value network.
    """
    def __init__(self, actor_net, value_net):
        super().__init__()
        self.actor_net = actor_net
        self.value_net = value_net
