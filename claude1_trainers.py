import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


# =============================================================================
# Token-Level KL Barrier: 核心改进
#
# 原始问题：你的 Proposition 证明了序列级 KL 求和存在"局部约束松弛"——
# 优化器可以在关键节点 t* 处集中消耗所有 KL 预算，
# 而将非关键节点的 KL 归零来摊薄总惩罚，使得序列级约束看似满足，
# 但局部语义节点已完全脱离参考分布。
#
# 原代码的问题：barrier 在 token 维度做了 .mean()，
# 等价于又引入了一次序列级摊薄，与要解决的问题同构。
#
# 修正思路：
#   对每个 token 的 KL 值 kl_t 直接施加 log-barrier 惩罚：
#       penalty_t = -mu * log(kappa - kl_t)   当 kl_t < kappa
#   其中 kappa 是 per-token KL 预算上界。
#   当某个 token 的 kl_t -> kappa 时，penalty_t -> +inf，
#   优化器无法通过其他 token 来摊薄这个惩罚，
#   因为惩罚是逐 token 独立计算的，再求和时高危节点的惩罚
#   已经趋于无穷，不可能被其他节点的零值稀释。
#
# 与 critic 的解耦：
#   原代码将 barrier 的强度依赖于 critic 的输出 v_safe，
#   但 critic 的标签本身存在语义反转问题（见下方 critic 注释）。
#   修正后 barrier 直接作用于 KL 值，与 critic 完全解耦，
#   避免了 critic 收敛前 barrier 无效的冷启动问题。
# =============================================================================

def _compute_token_kl_barrier(
    per_position_kl_rejected: torch.FloatTensor,   # [batch, seq_len]  每token KL值
    loss_mask_rejected: torch.FloatTensor,          # [batch, seq_len]  有效token掩码
    kappa: float = 2.0,                             # per-token KL预算上界
    mu: float = 0.1,                                # barrier强度系数
) -> torch.FloatTensor:
    """
    对 rejected 序列的每个 token 独立施加 log-barrier 惩罚。

    数学形式：
        penalty_t = -mu * log(kappa - kl_t)   if kl_t < kappa
                    +inf (clamp到大数)          if kl_t >= kappa

    这保证了：
    1. 当某 token 的 KL 接近预算上界 kappa 时，惩罚趋于无穷
    2. 各 token 的惩罚独立计算，高危节点 t* 无法被其他节点摊薄
    3. 最终对有效 token 求和（不做均值），保留 token 数量的量纲，
       使得含更多高危 token 的序列受到更强约束

    返回: [batch] 每个样本的 barrier 惩罚总量
    """
    # slack = kappa - kl_t，代表距预算上界的余量
    # clamp 到正数保证 log 有意义；趋近0时 log -> -inf，penalty -> +inf
    slack = (kappa - per_position_kl_rejected).clamp(min=1e-8)
    barrier_per_token = -mu * torch.log(slack)          # [batch, seq_len]

    # 仅对有效 token 施加惩罚，并求和（不做均值，保留惩罚量纲）
    # 这里用 sum 而非 mean：若用 mean，长序列中的高危 token 惩罚会被稀释
    barrier_loss = (barrier_per_token * loss_mask_rejected).sum(-1)  # [batch]
    return barrier_loss


# =============================================================================
# Safety Critic: 标签与监督信号修正
#
# 原代码问题：
#   target_v = torch.where(ref_margin > 0.0, -1.0, 1.0)
#   其中 ref_margin = ref_chosen_logps - ref_rejected_logps
#   当参考模型（helpfulness-aligned 的 pi_r）更偏好 chosen 时，ref_margin > 0，
#   此时 target 被设为 -1.0（危险）。
#
#   这个逻辑存在语义反转：
#   - 在 safety 数据集中，chosen = 安全回答，rejected = 不安全回答
#   - 参考模型 pi_r 是 helpfulness-aligned 的，它完全可能更偏好 rejected（不安全但流畅）
#   - 导致 ref_margin < 0，target = +1.0（安全）——但实际上 rejected 是不安全的
#   - 标签方向系统性错误
#
# 修正思路：
#   critic 的监督信号应直接来自数据集的 chosen/rejected 标注，
#   而不是参考模型的偏好边际。
#   - rejected token 的真实安全标签 = -1.0（这条数据就是不安全的）
#   - chosen token 的真实安全标签 = +1.0（这条数据是安全的）
#   这与数据集的标注语义完全对齐。
#
# critic 的作用重新定位：
#   修正后 critic 的主要作用不再是驱动 barrier（barrier 已独立），
#   而是作为一个 token 级安全评分器，用于：
#   1. 识别当前 batch 中哪些 token 位置是高危节点（辅助分析）
#   2. 通过自监督学习提供额外的正则信号，防止策略在安全方向过度漂移
#   critic_loss 的权重应小（当前 0.1），防止喧宾夺主
# =============================================================================

def _compute_critic_loss(
    v_safe_chosen: torch.FloatTensor,      # [batch, seq_len]  chosen序列的critic评分
    v_safe_rejected: torch.FloatTensor,    # [batch, seq_len]  rejected序列的critic评分
    loss_mask_chosen: torch.FloatTensor,   # [batch, seq_len]
    loss_mask_rejected: torch.FloatTensor, # [batch, seq_len]
) -> torch.FloatTensor:
    """
    使用数据集的 chosen/rejected 标注作为 critic 的直接监督信号。

    chosen  序列: 安全回答，target = +1.0
    rejected 序列: 不安全回答，target = -1.0

    用 smooth L1 loss（Huber loss）代替 MSE，对异常值更鲁棒，
    防止 critic 在训练初期因标签噪声导致梯度爆炸。

    返回: [batch] 每个样本的 critic loss（chosen 和 rejected 均值）
    """
    # chosen: target = +1.0（安全）
    target_chosen = torch.ones_like(v_safe_chosen)
    critic_loss_chosen = F.smooth_l1_loss(v_safe_chosen, target_chosen, reduction='none')
    critic_loss_chosen = (critic_loss_chosen * loss_mask_chosen).sum(-1) / (
        loss_mask_chosen.sum(-1) + 1e-8)

    # rejected: target = -1.0（不安全）
    target_rejected = -torch.ones_like(v_safe_rejected)
    critic_loss_rejected = F.smooth_l1_loss(v_safe_rejected, target_rejected, reduction='none')
    critic_loss_rejected = (critic_loss_rejected * loss_mask_rejected).sum(-1) / (
        loss_mask_rejected.sum(-1) + 1e-8)

    # 两项平均，保证 chosen 和 rejected 的梯度量级平衡
    return 0.5 * (critic_loss_chosen + critic_loss_rejected)


def tdpo_loss(
    chosen_logps_margin: torch.FloatTensor,
    rejected_logps_margin: torch.FloatTensor,
    chosen_position_kl: torch.FloatTensor,
    rejected_position_kl: torch.FloatTensor,
    token_kl_barrier_loss: torch.FloatTensor,   # [batch] token级KL barrier（已解耦）
    critic_loss: torch.FloatTensor,             # [batch] critic自监督损失（已修正标签）
    beta: float,
    alpha: float = 0.5,
    if_tdpo2: bool = True,
    critic_weight: float = 0.1,                 # critic权重调小，防止喧宾夺主
):
    """
    BSB-TDPO 损失函数。

    总损失 = TDPO2对齐损失
           + token级KL barrier（解决局部约束松弛）
           + critic自监督损失（辅助正则，权重较小）

    各项的梯度语义：
    - TDPO2项：驱动策略偏好 chosen > rejected，同时控制序列级KL
    - barrier项：对每个token的KL独立惩罚，防止高危节点t*处的KL爆炸
    - critic项：使 critic 正确区分安全/不安全 token，辅助信号
    """
    chosen_values = chosen_logps_margin + chosen_position_kl
    rejected_values = rejected_logps_margin + rejected_position_kl
    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if not if_tdpo2:
        # TDPO1: 双向SeqKL控制
        logits = chosen_rejected_logps_margin - (rejected_position_kl - chosen_position_kl)
    else:
        # TDPO2: stop-gradient 防止 chosen 端SeqKL加速
        logits = chosen_rejected_logps_margin - alpha * (
            rejected_position_kl - chosen_position_kl.detach()
        )

    # 主对齐损失
    align_loss = -F.logsigmoid(beta * logits)

    # 融合：token级barrier是主要约束项，critic是辅助正则项
    losses = align_loss + token_kl_barrier_loss + critic_weight * critic_loss

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()
    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor,
                     average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits."""
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def _tdpo_get_batch_logps(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
):
    """
    计算 TDPO 所需的 token 级统计量。

    相比原版新增返回值：
    - per_position_kl_raw: [batch, seq_len] 未经 mask/聚合的 token 级 KL，
      用于 token 级 barrier 的逐位计算
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    # per_position_kl shape: [batch, seq_len]，每个位置的 KL(ref || pi_theta)
    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2,
                                             index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        return (
            (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1),
            (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1),
            per_position_kl,   # [batch, seq_len] 原始token级KL，供barrier使用
            loss_mask,
        )
    else:
        return (
            (logps_margin * loss_mask).sum(-1),
            (per_position_kl * loss_mask).sum(-1),
            (per_token_logps * loss_mask).sum(-1),
            per_position_kl,   # [batch, seq_len] 原始token级KL，供barrier使用
            loss_mask,
        )


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor."""
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str,
                 reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """
        BSB-TDPO Trainer.

        相比原版的核心改动：
        1. safety_critic 同时处理 chosen 和 rejected 序列（原版只处理 rejected）
        2. token 级 KL barrier 通过 config 暴露 kappa/mu 两个超参
        3. critic 标签来自数据集标注，不再依赖参考模型偏好边际
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy

        # Safety Critic: 挂载线性探针
        #
        # LoRA 兼容说明：
        #   train_lora.py 中 policy 经过 get_peft_model() 包装为 PeftModel，
        #   PeftModel 不直接暴露 .config，需要通过 .base_model.model.config 访问。
        #   同时 safety_critic 是普通 nn.Linear，不在 LoRA 的 target_modules 里，
        #   get_peft_model 默认会冻结非 LoRA 参数，所以必须在挂载后
        #   手动将 critic 参数标记为 requires_grad=True。
        if not hasattr(self.policy, 'safety_critic'):
            # 兼容 PeftModel 和普通 nn.Module 两种情况
            base_cfg = (
                self.policy.base_model.model.config
                if hasattr(self.policy, 'base_model')
                else self.policy.config
            )
            hidden_size = base_cfg.hidden_size
            self.policy.safety_critic = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.policy.safety_critic.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.policy.safety_critic.bias)
            device = next(self.policy.parameters()).device
            self.policy.safety_critic = self.policy.safety_critic.to(device)

        # 确保 critic 参数可训练（LoRA 的 prepare_model_for_kbit_training
        # 和 get_peft_model 可能会冻结非 LoRA 参数）
        for param in self.policy.safety_critic.parameters():
            param.requires_grad = True

        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(
            **data_iterator_kwargs, split='train',
            n_epochs=config.n_epochs, n_examples=config.n_examples,
            batch_size=config.batch_size, silent=rank != 0,
            cache_dir=get_local_dir(config.local_dirs))
        rank0_print(f'Loaded train data iterator')

        self.eval_iterator = get_batch_iterator(
            **data_iterator_kwargs, split='test',
            n_examples=config.n_eval_examples,
            batch_size=config.eval_batch_size, silent=rank != 0,
            cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy for the given batch of inputs."""
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False,
                                               recurse=False) if 'FSDP' in self.config.trainer
                       else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'],
                attention_mask=batch['prompt_attention_mask'],
                max_length=self.config.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name == 'tdpo':
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False,
                                                   recurse=False) if 'FSDP' in self.config.trainer
                           else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'],
                    attention_mask=batch['prompt_attention_mask'],
                    max_length=self.config.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name == 'tdpo':
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def tdpo_concatenated_forward(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """
        BSB-TDPO 前向计算。

        核心改动说明：

        [改动1] _tdpo_get_batch_logps 新增返回 per_position_kl_raw 和 loss_mask
            原因：barrier 需要在 token 维度独立计算，必须拿到未聚合的 KL 值

        [改动2] token 级 KL barrier 替换原来的 critic-driven barrier
            原因：原 barrier 依赖 v_safe 的均值，退化为序列级摊薄；
                  新 barrier 直接在 kl_t 上作用，高危节点无法被摊薄

        [改动3] critic 同时处理 chosen 和 rejected
            原因：原版只对 rejected 建模，critic 对 chosen 分布盲目，
                  导致其评分器泛化能力不足

        [改动4] critic 标签来自数据集标注，而非 ref_margin
            原因：ref_margin 基于 helpfulness-aligned pi_r 的偏好，
                  在 safety 数据集上语义可能反转，导致标签方向系统性错误

        [改动5] 移除 is_safety_prompt soft-gating 对 barrier 的乘法门控
            原因：soft-gating 将 barrier 退化为样本级操作，
                  且 ref_margin 驱动的门控同样有语义反转风险；
                  barrier 应对所有 rejected token 一视同仁地施加约束

        超参来源（从 config 读取，方便消融实验）：
            config.loss.kappa: per-token KL 预算上界，默认 2.0
            config.loss.mu:    barrier 强度系数，默认 0.1
            config.loss.critic_weight: critic 损失权重，默认 0.1
        """
        concatenated_batch = concatenated_inputs(batch)

        # 策略模型前向，开启 hidden_states 供 critic 使用
        outputs = model(
            concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask'],
            output_hidden_states=True,
        )
        all_logits = outputs.logits.to(torch.float32)
        # hidden_states[-1]: [2*batch, seq_len, hidden_size]
        hidden_states = outputs.hidden_states[-1]

        # 参考模型前向，不需要梯度
        with torch.no_grad():
            reference_all_logits = reference_model(
                concatenated_batch['concatenated_input_ids'],
                attention_mask=concatenated_batch['concatenated_attention_mask'],
            ).logits.to(torch.float32)

        # [改动1] 新增返回 per_position_kl_raw 和 loss_mask
        (all_logps_margin,
         all_position_kl,
         all_logps,
         per_position_kl_raw,   # [2*batch, seq_len] 未聚合的token级KL
         loss_mask_all,         # [2*batch, seq_len] 有效token掩码
         ) = _tdpo_get_batch_logps(
            all_logits,
            reference_all_logits,
            concatenated_batch['concatenated_labels'],
            average_log_prob=False,
        )

        half = batch['chosen_input_ids'].shape[0]

        # 序列级聚合量（TDPO2主损失使用）
        chosen_logps_margin = all_logps_margin[:half]
        rejected_logps_margin = all_logps_margin[half:]
        chosen_position_kl = all_position_kl[:half]
        rejected_position_kl = all_position_kl[half:]
        chosen_logps = all_logps[:half].detach()
        rejected_logps = all_logps[half:].detach()

        # token 级 KL 和 mask（barrier 使用）
        per_position_kl_rejected_raw = per_position_kl_raw[half:]  # [batch, seq_len]
        loss_mask_rejected = loss_mask_all[half:].float()           # [batch, seq_len]
        loss_mask_chosen = loss_mask_all[:half].float()             # [batch, seq_len]

        # =====================================================================
        # [改动2] Token 级 KL Barrier
        #
        # 直接对每个 rejected token 的 KL 值施加 log-barrier：
        #   penalty_t = -mu * log(kappa - kl_t)
        # kl_t -> kappa 时 penalty_t -> +inf，无法被其他 token 摊薄。
        # 这直接对应你 Proposition 中 t* 节点的 KL 爆炸场景。
        # =====================================================================
        kappa = getattr(self.config.loss, 'kappa', 2.0)
        mu = getattr(self.config.loss, 'mu', 0.1)

        token_kl_barrier_loss = _compute_token_kl_barrier(
            per_position_kl_rejected_raw,
            loss_mask_rejected,
            kappa=kappa,
            mu=mu,
        )  # [batch]

        # =====================================================================
        # [改动3, 4] Safety Critic：同时处理 chosen/rejected，标签来自数据集标注
        #
        # v_safe shape: [2*batch, seq_len-1, 1] -> squeeze -> [2*batch, seq_len-1]
        # 注意 hidden_states 比 labels 多一个位置（含BOS），
        # 与 _tdpo_get_batch_logps 中对 labels[:, 1:] 的处理对齐。
        # =====================================================================
        v_safe_all = model.safety_critic(hidden_states[:, :-1, :]).squeeze(-1)
        # [2*batch, seq_len-1]

        v_safe_chosen = v_safe_all[:half]    # [batch, seq_len-1]
        v_safe_rejected = v_safe_all[half:]  # [batch, seq_len-1]

        critic_weight = getattr(self.config.loss, 'critic_weight', 0.1)
        critic_loss = _compute_critic_loss(
            v_safe_chosen,
            v_safe_rejected,
            loss_mask_chosen,
            loss_mask_rejected,
        )  # [batch]

        return (
            chosen_logps_margin,
            rejected_logps_margin,
            chosen_position_kl,
            rejected_position_kl,
            chosen_logps,
            rejected_logps,
            token_kl_barrier_loss,
            critic_loss,
            critic_weight,
            # 以下供 metrics 记录使用
            per_position_kl_rejected_raw.detach(),
            loss_mask_rejected,
        )

    def get_batch_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        loss_config: DictConfig,
        train: bool = True,
    ):
        """Compute the SFT or BSB-TDPO loss and metrics for the given batch."""
        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name == 'tdpo':
            (chosen_logps_margin,
             rejected_logps_margin,
             chosen_position_kl,
             rejected_position_kl,
             policy_chosen_logps,
             policy_rejected_logps,
             token_kl_barrier_loss,
             critic_loss,
             critic_weight,
             per_position_kl_rejected_raw,
             loss_mask_rejected,
             ) = self.tdpo_concatenated_forward(self.policy, self.reference_model, batch)

            losses, chosen_rewards, rejected_rewards = tdpo_loss(
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                token_kl_barrier_loss,
                critic_loss,
                beta=loss_config.beta,
                alpha=loss_config.alpha,
                if_tdpo2=loss_config.if_tdpo2,
                critic_weight=critic_weight,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (
                chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            all_device_chosen_position_kl = all_gather_if_needed(
                chosen_position_kl.detach(), self.rank, self.world_size)
            all_device_rejected_position_kl = all_gather_if_needed(
                rejected_position_kl.detach(), self.rank, self.world_size)

            metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/margin'] = (
                all_device_chosen_position_kl - all_device_rejected_position_kl
            ).cpu().numpy().tolist()

            # 新增：token级KL统计，用于监控局部约束松弛是否得到缓解
            # max_token_kl: 每个样本中 KL 最大的 token 的值，是观测 t* 节点的关键指标
            kl_rejected = per_position_kl_rejected_raw * loss_mask_rejected
            max_token_kl = kl_rejected.max(dim=-1).values
            all_device_max_token_kl = all_gather_if_needed(
                max_token_kl.detach(), self.rank, self.world_size)
            metrics[f'kl_{train_test}/max_token_kl_rejected'] = (
                all_device_max_token_kl.cpu().numpy().tolist())

            # barrier_loss 和 critic_loss 单独记录，便于消融分析
            all_device_barrier = all_gather_if_needed(
                token_kl_barrier_loss.detach(), self.rank, self.world_size)
            metrics[f'loss_{train_test}/token_kl_barrier'] = (
                all_device_barrier.cpu().numpy().tolist())

            all_device_critic = all_gather_if_needed(
                critic_loss.detach(), self.rank, self.world_size)
            metrics[f'loss_{train_test}/critic'] = all_device_critic.cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(
                batch['chosen_input_ids'],
                attention_mask=batch['chosen_attention_mask'],
            ).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(
                policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)
            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(
            policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or BSB-TDPO training, with periodic evaluation."""
        rank0_print(f'Using {self.config.optimizer} optimizer')

        # 构建参数组：
        #   LoRA 场景下 self.policy.parameters() 只包含 LoRA 参数（其余被冻结），
        #   safety_critic 是后挂的普通 nn.Linear，虽然 requires_grad=True，
        #   但 PeftModel.parameters() 不保证一定枚举到它（取决于 peft 版本）。
        #   因此显式构建两个参数组，确保 critic 一定进入优化器。
        if self.config.loss.name == 'tdpo' and hasattr(self.policy, 'safety_critic'):
            # critic 用更大的学习率加速收敛（线性层从随机初始化开始）
            critic_lr = getattr(self.config, 'critic_lr', self.config.lr * 5)
            param_groups = [
                {'params': [p for n, p in self.policy.named_parameters()
                            if 'safety_critic' not in n and p.requires_grad],
                 'lr': self.config.lr},
                {'params': list(self.policy.safety_critic.parameters()),
                 'lr': critic_lr},
            ]
            rank0_print(f'safety_critic lr = {critic_lr}')
        else:
            param_groups = [p for p in self.policy.parameters() if p.requires_grad]

        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            param_groups, lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name == 'tdpo':
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (
                    self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in 'tdpo':
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (
                        tqdm.tqdm(self.eval_batches, desc='Computing eval metrics')
                        if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(
                        eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(
                            local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(
                            f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples})'
                            f' < eval_batch_size ({self.config.eval_batch_size}).')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]

                    for eval_batch in (
                            tqdm.tqdm(sample_batches, desc='Generating samples...')
                            if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(
                            eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name == 'tdpo':
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(
                                    self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')

                if self.config.sample_during_eval:
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name == 'tdpo':
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)
                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name == 'tdpo':
                            wandb.log({"reference_samples": reference_text_table},
                                      step=self.example_counter)

                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(
                    batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(
                    global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(
                    local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(
                    f'train stats after {self.example_counter} examples: '
                    f'{formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(
                    f'skipping logging after {self.example_counter} examples '
                    f'to avoid logging too frequently')
            #### END TRAINING ####

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor],
                         metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')
        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""
        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics,
                              'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics,
                              'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics,
                              'scheduler.pt', output_dir)

        # 单独保存 safety_critic 权重到 run_dir 根目录，
        # 方便 train_lora.py 合并阶段找到并复制到 merged_final_model/
        if hasattr(self.policy, 'safety_critic'):
            critic_save_path = os.path.join(self.run_dir, 'safety_critic.pt')
            torch.save(self.policy.safety_critic.state_dict(), critic_save_path)
            rank0_print(f'safety_critic weights saved to {critic_save_path}')


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str,
                 reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """FSDP trainer: shards the model across multiple GPUs."""
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, \
            'must specify model.block_name for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={wrap_class})

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) \
            if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl)
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper, offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy,
                                               checkpoint_wrapper_fn=non_reentrant_wrapper,
                                               check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name == 'tdpo':
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip gradient norm for FSDP, gathering gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save policy and optimizer state, gathering from all processes."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT,
                                  state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict,
                                  metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT,
                                  optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict,
                                  metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict,
                                  metrics, 'scheduler.pt', output_dir)
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """TensorParallel trainer: shards the model across multiple GPUs."""
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)

        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name == 'tdpo':
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict,
                              metrics, 'policy.pt', output_dir)
        del policy_state_dict
