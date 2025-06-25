# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
import math

class DecoupledTopKGate(torch.nn.Module):
    """
    创新点：解耦门控网络 - 分离专家选择权重和输出融合权重
    Decoupled Gating Network that separates expert selection weights and output fusion weights
    """
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, 
                 momentum_factor=0.9, temperature=1.0, adaptive_noise=True, **options):
        super().__init__()
        
        # 专家选择网络 (Expert Selection Network)
        try:
            self.selection_gate = torch.nn.Linear(model_dim, num_global_experts, bias=False, 
                                                dtype=torch.float32 if fp32_gate else None)
        except:
            self.selection_gate = torch.nn.Linear(model_dim, num_global_experts, bias=False)
        
        # 输出融合网络 (Output Fusion Network) - 创新点
        try:
            self.fusion_gate = torch.nn.Linear(model_dim, num_global_experts, bias=False,
                                             dtype=torch.float32 if fp32_gate else None)
        except:
            self.fusion_gate = torch.nn.Linear(model_dim, num_global_experts, bias=False)
        
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate
        self.enable_softmax_logits = True
        self.num_global_experts = num_global_experts
        
        # 动量系统参数 (Momentum System Parameters) - 创新点
        self.momentum_factor = momentum_factor
        self.temperature = temperature
        self.adaptive_noise = adaptive_noise
        
        # 动量缓存 (Momentum Buffer)
        self.register_buffer('momentum_weights', torch.ones(num_global_experts))
        self.register_buffer('expert_usage_count', torch.zeros(num_global_experts))
        self.register_buffer('training_steps', torch.tensor(0))
        
        # 门控网络正则化
        self.normalize_gate = options.get('normalize_one_score_gate', False)
        if self.normalize_gate:
            print('Decoupled Gating: Normalizing gate vectors')

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise', 'normalize_one_score_gate', 
                          'momentum_factor', 'temperature', 'adaptive_noise'):
                raise Exception('Unrecognized argument provided to Decoupled Gating module: %s' % opt)

    def _update_momentum_weights(self, selected_experts):
        """更新动量权重系统"""
        if self.training:
            self.training_steps += 1
            
            # 统计专家使用频率
            expert_counts = torch.zeros(self.num_global_experts, device=selected_experts.device)
            for expert_id in selected_experts.view(-1):
                if expert_id < self.num_global_experts:
                    expert_counts[expert_id] += 1
            
            # 更新专家使用计数
            self.expert_usage_count = (self.momentum_factor * self.expert_usage_count + 
                                     (1 - self.momentum_factor) * expert_counts)
            
            # 计算动量权重 - 使用频率低的专家获得更高权重
            usage_normalized = self.expert_usage_count / (self.expert_usage_count.sum() + 1e-8)
            inverse_usage = 1.0 / (usage_normalized + 1e-8)
            self.momentum_weights = F.softmax(inverse_usage / self.temperature, dim=0)

    def _get_adaptive_noise(self):
        """自适应门控噪声 - 训练前期高噪声，后期低噪声"""
        if not self.adaptive_noise:
            return 1.0
        
        # 基于训练步数的噪声衰减
        progress = min(self.training_steps.float() / 10000.0, 1.0)  # 假设10000步为一个周期
        noise_factor = 2.0 * (1.0 - progress) + 0.1  # 从2.0衰减到0.1
        return noise_factor

    def forward(self, x):
        if self.normalize_gate:
            x = F.normalize(x, p=2, dim=-1)
        
        if self.fp32_gate:
            x = x.float()
            selection_gate = self.selection_gate.float()
            fusion_gate = self.fusion_gate.float()
        else:
            selection_gate = self.selection_gate
            fusion_gate = self.fusion_gate
        
        # 专家选择分数 (Expert Selection Scores)
        selection_scores = selection_gate(x)
        
        # 输出融合分数 (Output Fusion Scores) - 创新点
        fusion_scores = fusion_gate(x)
        
        # 应用动量权重到融合分数
        if self.training:
            fusion_scores = fusion_scores * self.momentum_weights.unsqueeze(0)
        
        # 自适应噪声
        if self.training and hasattr(self, 'gate_noise'):
            noise_factor = self._get_adaptive_noise()
            noise = torch.randn_like(selection_scores) * self.gate_noise * noise_factor
            selection_scores = selection_scores + noise
        
        # 选择top-k专家
        topk_values, topk_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        
        # 更新动量权重
        self._update_momentum_weights(topk_indices)
        
        # 返回选择分数和融合分数
        return {
            'selection_scores': selection_scores,
            'fusion_scores': fusion_scores,
            'topk_indices': topk_indices,
            'topk_values': topk_values
        }, self.top_k

Gate = DecoupledTopKGate
