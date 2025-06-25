# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
from torch import Tensor
from torch.nn import ModuleList
import torch.nn.functional as F
import numpy as np

from ..impls import communicate as C
from ..impls.generalized_fast_dispatch import fast_encode, fast_decode, extract_critical
from ..impls.overlap import a2a_ffn_overlap_forward
from . import losses

from ..gates.top import LinearTopKGate

class MOELayer(torch.nn.Module):
    """Tutel optimized MOELayer
    """
    @staticmethod
    def global_expert_count(num_local_experts, group=None):
        if not isinstance(num_local_experts, int):
            num_local_experts = -int(1 / (num_local_experts + 1e-5))
        world_size = C.get_world_size(group)
        if num_local_experts == 0:
            raise Exception("Invalid value of num_local_experts: %d" % num_local_experts)
        if num_local_experts > 0:
            return num_local_experts * world_size
        assert world_size % -num_local_experts == 0, f"Excepting {-num_local_experts} devices to share an expert param, while global device count is {world_size}."
        return world_size // -num_local_experts
    
    @staticmethod
    def local_expert_count(num_global_experts, group=None):       
        world_size = C.get_world_size(group)
        assert num_global_experts % world_size == 0, f"Excepting {num_global_experts} devices to share an expert param, while global device count is {world_size}."
        return num_global_experts // world_size

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buff_name = prefix + '_num_global_experts'
        if buff_name not in state_dict:
            logging.warning(f"\033[31mYou are loading a legacy format of checkpoint with at least one Tutel MoE layer inside, which wouldn't support new Tutel feature allowing the number of experts per checkpoint file to mutate.\033[0m")
            logging.warning(f"\033[31m  The next time you overwrite it with new checkpoint, the recording format will be updated automatically.\033[0m")
            logging.warning(f"\033[31m  However, the new format won't be compatible with early Tutel versions, unless you force loading it with `model.load_state_dict(.., strict=False)`.\033[0m")
            state_dict[buff_name] = self._num_global_experts
        else:
            state_experts, expect_experts = int(state_dict[buff_name]), self.num_global_experts
            # assert state_experts == expect_experts, "Failed to load state from checkpoint: the number of global experts mismatch (%s <- %s)" % (expect_experts, state_experts)

        buff_name = prefix + '_max_num_global_experts'
        if buff_name not in state_dict:
            logging.warning(f"\033[31mYou are loading a legacy format of checkpoint with at least one Tutel MoE layer inside, which wouldn't support new Tutel feature allowing the number of experts per checkpoint file to mutate.\033[0m")
            logging.warning(f"\033[31m  The next time you overwrite it with new checkpoint, the recording format will be updated automatically.\033[0m")
            logging.warning(f"\033[31m  However, the new format won't be compatible with early Tutel versions, unless you force loading it with `model.load_state_dict(.., strict=False)`.\033[0m")
            state_dict[buff_name] = self._max_num_global_experts
        else:
            state_experts, expect_experts = int(state_dict[buff_name]), self.max_num_global_experts
            assert state_experts == expect_experts, "Failed to load state from checkpoint: the max number of global experts mismatch (%s <- %s)" % (expect_experts, state_experts)

        for name, param in self.experts.named_parameters():
            buff_name = prefix + 'experts.' + name
            assert buff_name in state_dict, "Could not find parameter `%s` in state_dict." % buff_name
            if state_dict[buff_name].numel() == param.numel():
                state_dict[buff_name] = state_dict[buff_name].view(param.shape)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)

    @property
    def num_global_experts(self):
        return int(self._num_global_experts)
    
    @property
    def max_num_global_experts(self):
        return int(self._max_num_global_experts)

    def __init__(
        self,
        gate_type,
        model_dim: int,
        experts=None,
        scan_expert_func=None,
        result_func=None,
        group=None,
        seeds=None,
        a2a_ffn_overlap_degree=1,
        is_postscore=True,
        batch_prioritized_routing=False,
        normalize_gate=True,
        is_gshard_loss=True,
        parallel_type='adaptive:1',
        use_2dh=False,
        one_score_gate=False,
        normalize_one_score_gate=False,
        value_norm_weighted=False,
        update_momentum=0.0,
        # DDG-ALS 创新参数
        enable_ddg_als=False,  # 启用动态解耦门控和自适应损失调度
        decoupled_gating=False,  # 解耦门控开关
        adaptive_loss_schedule=False,  # 自适应损失调度开关
        expert_momentum=0.9,  # 专家激活动量
        temperature_init=1.0,  # 初始温度
        temperature_decay=0.995,  # 温度衰减率
        exploration_bonus_init=0.1,  # 初始探索奖励
        exploration_decay=0.99,  # 探索奖励衰减
        loss_schedule_warmup=1000,  # 损失调度预热步数
        # share_value=False,
        **kwargs
    ):
        super().__init__()
        assert model_dim % 2 == 0, "Model_dim (%s) must be even value, while this Model_dim mod 2 > 0." % model_dim
        group = group or dist.group.WORLD

        if 'pad_samples' in kwargs:
            logging.warning(f"`pad_samples` option in Tutel Moe-layer has been deprecated, as Tutel always assumes `pad_samples=False` for better efficiency.")
            kwargs.pop('pad_samples')
        for k in kwargs:
            raise Exception('Unrecognized argument provided to Tutel Moe-layer: %s' % k)

        self.group = group
        self.result_func = result_func
        self.skip_moe = (int(os.environ.get('SKIP_MOE', '0')) != 0)

        # initially set max number equal to local number
        self.num_local_experts = experts.pop('count_per_node', 1)
        self.max_num_local_experts = self.num_local_experts
        self.register_buffer('_num_global_experts', torch.tensor(MOELayer.global_expert_count(self.num_local_experts, self.group)))
        self.register_buffer('_max_num_global_experts', torch.tensor(MOELayer.global_expert_count(self.max_num_local_experts, self.group)))

        self.world_size = C.get_world_size(self.group)
        # in fact this will not happen
        if self.num_global_experts < self.world_size:
            self.sharded_count = self.world_size // self.num_global_experts
            self.num_local_experts = 1
        else:
            self.sharded_count = 1

        self.auto_parallel, self.adaptive_degree, self.use_model_parallel = False, self.sharded_count, True
        self.valid_rs = [0] + [i for i in range(1, self.sharded_count + 1) if self.sharded_count % i == 0]
        # 实际上valid_rs总是[0]

        if parallel_type.startswith('adaptive:'):
            self.adaptive_degree = int(parallel_type[parallel_type.index(':') + 1:])
            self.adaptive_degree = min(max(self.adaptive_degree, 0), self.sharded_count)
            if self.adaptive_degree not in self.valid_rs:
                raise Exception("Unexpected value of adaptive_degree: %d, expecting a candidate within %s." % (self.adaptive_degree, self.valid_rs))
        elif self.sharded_count == 1:
            pass
        elif parallel_type in ('data', 'model'):
            self.adaptive_degree = 1 if parallel_type == 'data' else self.sharded_count
        elif parallel_type == 'auto':
            self.adaptive_degree = 1
        else:
            raise Exception('Unrecognized parallel type specified: %s' % parallel_type)

        self.model_dim = model_dim

        self.is_postscore = is_postscore
        self.batch_prioritized_routing = batch_prioritized_routing
        if int(os.environ.get('BATCH_PRIO', 0)) != 0:
            self.batch_prioritized_routing = True
        self.normalize_gate = normalize_gate
        self.is_gshard_loss = is_gshard_loss

        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2dh = use_2dh

        if seeds is not None and seeds[1] is not None:
            torch.manual_seed(seeds[1])
            
        # self.share_value = share_value

        # GAMoE requires define the gates first.
        if isinstance(gate_type, str) and (not gate_type.startswith("GMGate")):
            
            assert re.match(r'^Top[0-9]+Gate$', gate_type), "Unrecognized gate_type: %s" % gate_type
            top_k = int(gate_type[3:-4])
            logging.warning(f"gate_type value `{gate_type}` in Tutel Moe-layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")
            gate_type = {'type': 'top', 'k': top_k}

        elif isinstance(gate_type, str) and gate_type.startswith("GMGate"):
            max_K = int(gate_type[6:])
            gate_type = {'type': 'gated_multi_gate', 'max_expert_num': max_K}
            # if use GAMoE, set the max number of global experts to the parameters in config
            
        if 'max_expert_num' in gate_type:
            self._max_num_global_experts.data = torch.tensor(int(gate_type['max_expert_num']))
            self.max_num_local_experts = self.local_expert_count(self.max_num_global_experts, self.group)

        if not isinstance(gate_type, list):
            gate_type = [gate_type]

        self.gates = []
        for gi, single_gate_type in enumerate(gate_type):
            gate_type = single_gate_type['type']
            single_gate_type.pop('type')
            assert re.match(r'[a-zA-Z0-9\_]+', gate_type), "Gate type must only include digits, letters and underline characters."

            if seeds is not None and seeds[0] is not None:
                torch.manual_seed(seeds[0] + gi)
            try:
                single_gate = importlib.import_module(f'...gates.{gate_type}', __name__)
            except ModuleNotFoundError:
                raise Exception("Unrecognized gate_type: %s" % gate_type)

            gate_module = single_gate.Gate(model_dim=self.model_dim, num_global_experts=self.num_global_experts, normalize_one_score_gate=normalize_one_score_gate, **single_gate_type)
            if not hasattr(gate_module, 'gate_noise'):
                gate_module.gate_noise = single_gate_type.get('gate_noise', 0.0)
            if not hasattr(gate_module, 'capacity_factor'):
                gate_module.capacity_factor = single_gate_type.get('capacity_factor', float(os.environ.get('CAP_FACTOR', 1.0)))

            self.gates += [gate_module]
        print("Gate types: ", [x.__class__.__name__ for x in self.gates])
        self.gates = ModuleList(self.gates)

        experts_type = experts.pop('type')
        if experts_type == 'custom':
            self.experts = cast(ModuleList, experts['module'])
        else:
            assert re.match(r'[a-zA-Z0-9\_]+', experts_type), "Expert type must only include digits, letters and underline characters."
            try:
                fused_experts = importlib.import_module(f'...experts.{experts_type}', __name__)
            except ModuleNotFoundError:
                raise Exception('Builtin expert type is not recognized: %s' % experts_type)

            if experts_type == 'ffn':
                assert 'fused_custom_fn' not in experts, "`fused_custom_fn` option for Tutel Moe-layer has been deprecated, please follows helloworld_from_scratch.py for custom construction instead."
                assert 'implicit_dropout_p' not in experts, "`implicit_dropout_p` option for Tutel Moe-layer has been deprecated, please use torch.nn.Dropout(p=implicit_dropout_p) on custom activation_fn (for fc1_dropout) and after Tutel Moe-layer (for fc2_dropout) instead."

            self.experts = fused_experts.ExpertModule(**experts)

        self.experts.update(self)

        if scan_expert_func is not None:
            for n, p in self.experts.named_parameters():
                scan_expert_func(n, p)
        for n, p in self.experts.named_parameters():
            setattr(p, '_tutel_expert', True)

        
        #专门用于EMoE
        self.one_score_gate = one_score_gate
        if self.one_score_gate:
            self.update_momentum = update_momentum
            assert isinstance(self.gates[0], LinearTopKGate), "only simple gate is supported"
            print("Using one score gate with momentum {}, freeze gate weight!".format(self.update_momentum))
            self.normalize_one_score_gate = normalize_one_score_gate
            self.gates[0].wg.weight.require_grad = False
            self.value_norm_weighted = value_norm_weighted
            if self.value_norm_weighted:
                self.register_buffer('expert_importance', torch.ones(self.num_global_experts) / self.num_global_experts)
        
        # DDG-ALS 创新机制初始化
        self.enable_ddg_als = enable_ddg_als
        self.decoupled_gating = decoupled_gating
        self.adaptive_loss_schedule = adaptive_loss_schedule
        
        if self.enable_ddg_als:
            print(f"🚀 Initializing DDG-ALS: Decoupled Gating={decoupled_gating}, Adaptive Loss={adaptive_loss_schedule}")
            
            # 动态解耦门控相关参数
            if self.decoupled_gating:
                # 专家选择权重 (用于路由)
                self.selection_gate = torch.nn.Linear(model_dim, self.max_num_global_experts, bias=False)
                # 输出融合权重 (用于加权输出，基于动量更新)
                self.register_buffer('fusion_weights', torch.ones(self.max_num_global_experts) / self.max_num_global_experts)
                self.expert_momentum = expert_momentum
                print(f"  ✓ Decoupled gating initialized with momentum {expert_momentum}")
            
            # 自适应温度机制
            self.register_buffer('current_temperature', torch.tensor(temperature_init))
            self.temperature_decay = temperature_decay
            
            # 专家激活历史跟踪
            self.register_buffer('expert_activation_history', torch.zeros(self.max_num_global_experts))
            self.register_buffer('expert_usage_count', torch.zeros(self.max_num_global_experts))
            
            # 自适应损失调度参数
            if self.adaptive_loss_schedule:
                self.register_buffer('training_step', torch.tensor(0))
                self.exploration_bonus_init = exploration_bonus_init
                self.exploration_decay = exploration_decay
                self.loss_schedule_warmup = loss_schedule_warmup
                self.register_buffer('current_aux_weight', torch.tensor(0.001))  # 初始很小的辅助损失
                print(f"  ✓ Adaptive loss scheduling initialized with warmup {loss_schedule_warmup} steps")
        
        self.record_routing = False
        
        if seeds is not None and len(seeds) > 2 and seeds[2] is not None:
            torch.manual_seed(seeds[2])

    def extra_repr(self):
        return 'Top-K(s) = %s, Total-Experts = %d [managed by %d device(s)],' % (
            [f'k={x.top_k}, noise={x.gate_noise}' for x in self.gates],
            self.num_global_experts,
            self.world_size,
        )

    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gates.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

    def expert_local(self, x, reserve_shape):
        y = self.experts(x.view(x.size(0), x.size(1), *reserve_shape), self)
        self.protected_shape = y.shape
        return y.reshape(y.size(0), y.size(1), -1)

    def begin_record_routing(self):
        self.reset_record_routing()
        self.record_routing = True

    def end_record_routing(self):
        self.record_routing = False
    
    def reset_record_routing(self):
        # self.routing_records = torch.zeros(self.num_global_experts, dtype=torch.long, device=self.experts.batched_fc1_w.device)
        self.routing_records = torch.zeros(self.max_num_global_experts + 1, dtype=torch.long, device=self.experts.batched_fc1_w.device)
        self.sample_records = None

    def get_routing_records(self):
        return self.routing_records[:self.max_num_global_experts]
    
    def get_sample_records(self):
        return self.sample_records
    
    def dynamic_decoupled_gating(self, x, gate_logits):
        """
        动态解耦门控机制：将专家选择与输出融合权重解耦
        """
        if not self.decoupled_gating:
            return gate_logits, None
            
        batch_size, seq_len, model_dim = x.shape
        x_flat = x.view(-1, model_dim)
        
        # 1. 专家选择权重 (用于路由决策)
        selection_logits = self.selection_gate(x_flat)
        selection_logits = selection_logits / self.current_temperature  # 温度缩放
        
        # 2. 基于历史激活模式的动态调整
        expert_diversity_bonus = self._compute_diversity_bonus()
        selection_logits = selection_logits + expert_diversity_bonus.unsqueeze(0)
        
        # 3. 输出融合权重基于动量更新
        current_selection = F.softmax(selection_logits, dim=-1)
        expert_activation = current_selection.mean(dim=0)  # 当前批次的专家激活率
        
        # 动量更新融合权重
        self.fusion_weights.data = (self.expert_momentum * self.fusion_weights + 
                                   (1 - self.expert_momentum) * expert_activation)
        
        # 4. 更新专家激活历史
        self.expert_activation_history.data = (0.95 * self.expert_activation_history + 
                                              0.05 * expert_activation)
        self.expert_usage_count.data += (expert_activation > 0.01).float()
        
        return selection_logits, self.fusion_weights
    
    def _compute_diversity_bonus(self):
        """
        计算专家多样性奖励，鼓励使用不活跃的专家
        """
        # 基于使用频率的逆向奖励
        usage_normalized = self.expert_usage_count / (self.expert_usage_count.sum() + 1e-8)
        diversity_bonus = -torch.log(usage_normalized + 1e-8) * 0.01
        
        # 为从未使用的专家提供额外奖励
        never_used_mask = (self.expert_usage_count == 0)
        diversity_bonus[never_used_mask] += 0.1
        
        return diversity_bonus
    
    def adaptive_loss_scheduling(self, base_aux_loss):
        """
        自适应损失调度：训练早期鼓励探索，后期保证稳定
        """
        if not self.adaptive_loss_schedule:
            return base_aux_loss
            
        step = self.training_step.item()
        
        # 1. 计算当前训练阶段的损失权重
        if step < self.loss_schedule_warmup:
            # 早期：低辅助损失 + 探索奖励
            progress = step / self.loss_schedule_warmup
            aux_weight = 0.001 + 0.009 * progress  # 从0.001逐渐增加到0.01
            
            # 探索奖励：鼓励专家分化
            exploration_bonus = self.exploration_bonus_init * (self.exploration_decay ** step)
            diversity_loss = self._compute_diversity_loss() * exploration_bonus
            
            total_loss = base_aux_loss * aux_weight + diversity_loss
            
        else:
            # 后期：标准辅助损失权重，注重稳定性
            aux_weight = 0.01 + 0.01 * min(1.0, (step - self.loss_schedule_warmup) / 10000)  # 最高到0.02
            stability_loss = self._compute_stability_loss() * 0.005
            
            total_loss = base_aux_loss * aux_weight + stability_loss
        
        # 更新当前辅助损失权重（用于记录）
        self.current_aux_weight.data = torch.tensor(aux_weight)
        
        return total_loss
    
    def _compute_diversity_loss(self):
        """
        计算多样性损失，鼓励专家分化
        """
        # 专家激活的方差越大越好（鼓励分化）
        activation_var = torch.var(self.expert_activation_history)
        diversity_loss = -torch.log(activation_var + 1e-8)  # 负对数，方差越大损失越小
        
        # 专家使用平衡性
        uniform_target = torch.ones_like(self.expert_activation_history) / len(self.expert_activation_history)
        balance_loss = F.kl_div(F.log_softmax(self.expert_activation_history, dim=0), 
                               uniform_target, reduction='batchmean')
        
        return diversity_loss + balance_loss
    
    def _compute_stability_loss(self):
        """
        计算稳定性损失，确保训练后期的稳定性
        """
        # 专家激活率的稳定性
        target_activation = 1.0 / self.num_global_experts
        stability_loss = F.mse_loss(self.expert_activation_history, 
                                   torch.full_like(self.expert_activation_history, target_activation))
        
        # 防止专家激活率过于集中
        max_activation = torch.max(self.expert_activation_history)
        concentration_penalty = F.relu(max_activation - 2 * target_activation) ** 2
        
        return stability_loss + concentration_penalty
    
    def update_temperature(self):
        """
        更新门控温度，实现自适应温度调节
        """
        if self.enable_ddg_als:
            self.current_temperature.data *= self.temperature_decay
            self.current_temperature.data = torch.clamp(self.current_temperature.data, min=0.1, max=5.0)
    
    def get_ddg_als_stats(self):
        """
        获取DDG-ALS统计信息，用于监控和调试
        """
        if not self.enable_ddg_als:
            return {}
            
        stats = {
            'training_step': self.training_step.item(),
            'current_temperature': self.current_temperature.item(),
            'current_aux_weight': self.current_aux_weight.item(),
            'expert_activation_std': torch.std(self.expert_activation_history).item(),
            'expert_max_activation': torch.max(self.expert_activation_history).item(),
            'expert_min_activation': torch.min(self.expert_activation_history).item(),
        }
        
        if self.decoupled_gating:
            stats.update({
                'fusion_weights_std': torch.std(self.fusion_weights).item(),
                'fusion_weights_max': torch.max(self.fusion_weights).item(),
                'fusion_weights_min': torch.min(self.fusion_weights).item(),
            })
            
        return stats
