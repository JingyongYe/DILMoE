# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F

def _one_hot_with_dtype(data, num_classes, dtype, hot_value=1):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), hot_value)
    return result

def _one_hot_with_dtype_and_padding(data, num_classes, dtype, hot_value=1):
    result = torch.zeros([data.size(0), num_classes + 1], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), hot_value)
    return result[:, :-1]

def empty_loss(scores_w_noise, top_ids):
    return 0.0

def sparse_loss(scores_w_noise, top_ids, num_global_experts):
    num_samples, max_num_global_experts = int(scores_w_noise.size(0)), int(scores_w_noise.size(1))
    topk_prob = torch.sum(scores_w_noise, dim=1) / (num_global_experts + 0.0)

    return torch.mean(topk_prob)

def diverse_and_simple_gate_loss(scores_w_noise, top_ids, gates, expert_mask):
    sims = torch.matmul(F.normalize(gates, dim=0).T, F.normalize(gates, dim=0))

    targets = torch.eye(sims.shape[0]).to(sims.device)

    sim_mask = torch.matmul(expert_mask.unsqueeze(0).T, expert_mask.unsqueeze(0))
    # sim_mask = sim_mask * (1.0 - torch.eye(sim_mask.shape[0]).to(sim_mask.device))

    # sims = sims * sim_mask

    sim_loss = torch.norm(sims * sim_mask - targets * sim_mask)
    # sim_loss = torch.sum(sims)

    simple_loss = torch.mean(torch.norm(gates, dim=0))

    return sim_loss + simple_loss





def gshard_loss(scores_w_noise, top_ids, num_global_experts):
    num_samples, max_num_global_experts = int(scores_w_noise.size(0)), int(scores_w_noise.size(1))
    mask = _one_hot_with_dtype_and_padding(top_ids[:, 0], max_num_global_experts, dtype=scores_w_noise.dtype,
        hot_value=num_global_experts / num_samples)
    me = torch.sum(scores_w_noise, dim=0)
    ce = torch.sum(mask, dim=0)
    l_aux = torch.sum(me * ce) / num_samples
    return l_aux

def load_importance_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
    def load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
        assert gate_noise > 0, "`gate_noise` must be > 0 for normalization in load_importance_loss()."
        normal = Normal(
            torch.tensor([0.0], device=scores_wo_noise.device),
            torch.tensor([gate_noise / num_global_experts], device=scores_wo_noise.device),
        )
        threshold = topk_logits[:, -1].view(-1, 1).float()
        diff = scores_wo_noise.float() - threshold.float()
        prob = normal.cdf(diff)
        Load = prob.sum(0)
        l_load = Load.float().var() / (Load.float().mean() ** 2 + 1e-10)
        return l_load

    def importance_loss(scores_wo_noise):
        Impi = scores_wo_noise.float().sum(0)
        l_imp = Impi.float().var() / (Impi.float().mean() ** 2 + 1e-10)

        return l_imp

    l_imp = importance_loss(scores_wo_noise)
    l_load = load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise)
    return (l_imp + l_load) / 2.0

def progressive_auxiliary_loss(scores_w_noise, top_ids, num_global_experts, 
                             training_step=0, total_steps=100000, 
                             min_weight=0.001, max_weight=0.1):
    """
    创新点：渐进式辅助损失策略
    Progressive Auxiliary Loss that starts small to encourage expert diversification,
    then increases to ensure load balancing in later training stages.
    
    Args:
        scores_w_noise: Gate scores with noise
        top_ids: Selected expert indices
        num_global_experts: Total number of experts
        training_step: Current training step
        total_steps: Total training steps
        min_weight: Minimum loss weight (early training)
        max_weight: Maximum loss weight (late training)
    """
    # 计算训练进度 (Training Progress)
    progress = min(training_step / total_steps, 1.0)
    
    # 渐进式权重策略 (Progressive Weight Strategy)
    # 前期小权重鼓励分化，后期大权重保证均衡
    if progress < 0.3:  # 前30%训练时间：鼓励分化
        loss_weight = min_weight + (max_weight - min_weight) * (progress / 0.3) * 0.2
    elif progress < 0.7:  # 中间40%训练时间：平缓过渡
        loss_weight = min_weight + (max_weight - min_weight) * 0.5
    else:  # 后30%训练时间：强制均衡
        loss_weight = min_weight + (max_weight - min_weight) * (0.5 + 0.5 * ((progress - 0.7) / 0.3))
    
    # 基础负载均衡损失
    base_loss = gshard_loss(scores_w_noise, top_ids, num_global_experts)
    
    # 专家多样性损失 (Expert Diversity Loss) - 创新点
    diversity_loss = 0.0
    if progress < 0.5:  # 前期更注重多样性
        num_samples = scores_w_noise.size(0)
        expert_probs = torch.sum(scores_w_noise, dim=0) / num_samples
        # 鼓励均匀分布，但允许一定程度的不均匀
        diversity_target = torch.ones_like(expert_probs) / num_global_experts
        diversity_loss = F.kl_div(expert_probs.log(), diversity_target, reduction='batchmean')
        diversity_weight = (0.5 - progress) * 2.0  # 权重随进度递减
        diversity_loss = diversity_loss * diversity_weight
    
    return loss_weight * base_loss + diversity_loss


def decoupled_gate_loss(selection_scores, fusion_scores, top_ids, num_global_experts,
                       expert_mask=None, orthogonal_weight=0.01):
    """
    创新点：解耦门控损失
    Loss function for decoupled gating that encourages orthogonality between
    selection and fusion networks while maintaining load balancing.
    """
    # 基础负载均衡损失
    base_loss = gshard_loss(selection_scores, top_ids, num_global_experts)
    
    # 选择-融合正交损失 (Selection-Fusion Orthogonality Loss) - 创新点
    if hasattr(selection_scores, 'grad_fn') and hasattr(fusion_scores, 'grad_fn'):
        # 计算选择和融合分数的相关性
        selection_norm = F.normalize(selection_scores, p=2, dim=1)
        fusion_norm = F.normalize(fusion_scores, p=2, dim=1)
        correlation = torch.sum(selection_norm * fusion_norm, dim=1)
        
        # 鼓励正交性（相关性接近0）
        orthogonal_loss = torch.mean(correlation ** 2)
        
        return base_loss + orthogonal_weight * orthogonal_loss
    
    return base_loss


def adaptive_importance_loss(scores_wo_noise, topk_logits, num_global_experts, 
                           gate_noise, training_step=0, total_steps=100000):
    """
    创新点：自适应重要性损失
    Adaptive importance loss that adjusts based on training progress.
    """
    # 原始重要性损失
    base_loss = load_importance_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise)
    
    # 训练进度自适应权重
    progress = min(training_step / total_steps, 1.0)
    
    # 早期训练：较小的重要性损失权重，允许专家自由分化
    # 后期训练：较大的重要性损失权重，强制负载均衡
    adaptive_weight = 0.1 + 0.9 * (progress ** 2)  # 二次增长
    
    return adaptive_weight * base_loss