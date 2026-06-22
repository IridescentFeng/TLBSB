#!/usr/bin/env python3
"""
TLBSB-v2 Sanity Check
验证 c = W_lm @ w_s 的词表安全信号质量——v2 能否立项的地基检验。

用法:
  python sanity_check_v2.py \
      --model_path /home/zjq/FZ2026/BSB-TDPO/alpaca-7b \
      --ckpt_path  /path/to/V6_run/step-XXXXX/policy.pt

只用 CPU，不占 GPU，约 1-2 分钟。
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── 有语义内容的 token 判定 ────────────────────────────────────────────────────
_STOPWORDS = {
    'the','a','an','is','are','was','were','be','been','being',
    'of','in','to','and','or','but','if','that','this','it',
    'i','you','we','he','she','they','my','your','our','their',
    'have','has','had','do','does','did','will','would','can','could',
    'should','may','might','shall','must','about','with','for','from',
    'by','at','on','up','as','so','not','no','all','one','out',
}

def is_semantic(tok: str) -> bool:
    t = tok.replace('▁','').replace('Ġ','').replace('##','').strip().lower()
    if len(t) < 3:
        return False
    if t.isdigit():
        return False
    if not any(c.isalpha() for c in t):
        return False
    if t in _STOPWORDS:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='TLBSB-v2 词表安全信号 sanity check'
    )
    parser.add_argument('--model_path', required=True,
                        help='底座模型路径（tokenizer + lm_head 来源）')
    parser.add_argument('--ckpt_path', required=True,
                        help='V6 policy.pt 路径（含 safety_critic.weight）')
    parser.add_argument('--topk', type=int, default=30,
                        help='展示 top/bottom k 个 token（默认 30）')
    parser.add_argument('--list_keys', action='store_true',
                        help='只打印 checkpoint 的所有 key，用于调试')
    args = parser.parse_args()

    # ── 1. tokenizer ───────────────────────────────────────────────────────────
    print(f'\n[1/4] 加载 tokenizer: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f'      词表大小: {tokenizer.vocab_size}')

    # ── 2. checkpoint → critic 权重 ────────────────────────────────────────────
    print(f'\n[2/4] 加载 checkpoint: {args.ckpt_path}')
    ckpt = torch.load(args.ckpt_path, map_location='cpu')

    # trainers.py 用 {'step_idx', 'state', 'metrics'} 结构保存
    state_dict = ckpt.get('state', ckpt)

    if args.list_keys:
        print('Checkpoint keys:')
        for k, v in state_dict.items():
            shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v)
            print(f'  {k:60s}  {shape}')
        sys.exit(0)

    # 找 critic weight（支持可能的前缀差异）
    critic_key = next(
        (k for k in state_dict if k.endswith('safety_critic.weight')),
        None
    )
    if critic_key is None:
        candidates = [k for k in state_dict if 'critic' in k or 'safety' in k]
        print(f'[ERROR] 没找到 safety_critic.weight。')
        print(f'        含 critic/safety 的 key: {candidates}')
        print(f'        用 --list_keys 查看所有 key。')
        sys.exit(1)

    w_s = state_dict[critic_key].squeeze().float()  # [H]
    print(f'      critic key : {critic_key}')
    print(f'      w_s shape  : {tuple(w_s.shape)}')
    print(f'      w_s norm   : {w_s.norm():.4f}')

    # ── 3. lm_head 权重 ────────────────────────────────────────────────────────
    print(f'\n[3/4] 获取 lm_head 权重')

    # LoRA checkpoint 一般不存 lm_head，先检查
    lm_key = next(
        (k for k in state_dict
         if 'lm_head.weight' in k or 'embed_out.weight' in k),
        None
    )

    if lm_key:
        print(f'      从 checkpoint 直接读取: {lm_key}')
        lm_W = state_dict[lm_key].float()
    else:
        print(f'      checkpoint 无 lm_head，从底座加载（float16，CPU）')
        base = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map='cpu',
            low_cpu_mem_usage=True,
        )
        lm_W = base.get_output_embeddings().weight.detach().float()  # [V, H]
        del base

    print(f'      lm_W shape : {tuple(lm_W.shape)}')

    # 维度匹配检查
    if lm_W.shape[1] != w_s.shape[0]:
        print(f'[ERROR] 维度不匹配: lm_W=[{lm_W.shape}], w_s=[{w_s.shape}]')
        print('        请确认 ckpt 和 model_path 来自同一底座。')
        sys.exit(1)

    # ── 4. 投影 & 排序 ─────────────────────────────────────────────────────────
    print(f'\n[4/4] 计算 c = lm_W @ w_s ...')
    c = (lm_W @ w_s).float()                         # [V]

    topk_res = c.topk(args.topk)
    botk_res = (-c).topk(args.topk)

    top_ids    = topk_res.indices.tolist()
    bot_ids    = botk_res.indices.tolist()
    top_scores = topk_res.values.tolist()
    bot_scores = [-v for v in botk_res.values.tolist()]  # 还原为实际负值

    top_tokens = tokenizer.convert_ids_to_tokens(top_ids)
    bot_tokens = tokenizer.convert_ids_to_tokens(bot_ids)

    # ── 打印 top ───────────────────────────────────────────────────────────────
    W = 62
    print(f'\n{"="*W}')
    print(f'  最安全 top-{args.topk}  (c_a 最大 → 生成此 token 朝"安全"方向)')
    print(f'{"="*W}')
    print(f'  {"token":<22}  {"id":>6}  {"score":>10}  semantic?')
    print(f'  {"-"*22}  {"-"*6}  {"-"*10}  ---------')
    for tok, idx, sc in zip(top_tokens, top_ids, top_scores):
        sem = '✓' if is_semantic(tok) else ' '
        print(f'  {tok:<22}  {idx:>6}  {sc:>+10.4f}  {sem}')

    print(f'\n{"="*W}')
    print(f'  最有害 bot-{args.topk}  (c_a 最小 → 生成此 token 朝"有害"方向)')
    print(f'{"="*W}')
    print(f'  {"token":<22}  {"id":>6}  {"score":>10}  semantic?')
    print(f'  {"-"*22}  {"-"*6}  {"-"*10}  ---------')
    for tok, idx, sc in zip(bot_tokens, bot_ids, bot_scores):
        sem = '✓' if is_semantic(tok) else ' '
        print(f'  {tok:<22}  {idx:>6}  {sc:>+10.4f}  {sem}')

    # ── 诊断 ──────────────────────────────────────────────────────────────────
    top_sem_n = sum(1 for t in top_tokens if is_semantic(t))
    bot_sem_n = sum(1 for t in bot_tokens if is_semantic(t))

    print(f'\n{"="*W}')
    print(f'  诊断摘要')
    print(f'{"="*W}')
    print(f'  得分统计: mean={c.mean():+.4f}  std={c.std():.4f}  '
          f'min={c.min():+.4f}  max={c.max():+.4f}')
    print(f'  top-{args.topk} 有语义 token: {top_sem_n}/{args.topk} '
          f'({100*top_sem_n/args.topk:.0f}%)')
    print(f'  bot-{args.topk} 有语义 token: {bot_sem_n}/{args.topk} '
          f'({100*bot_sem_n/args.topk:.0f}%)')
    print()

    if bot_sem_n >= 18:
        verdict = '✅  地基稳'
        advice  = ('bot-30 语义 token 充足，c_a 信号有意义。\n'
                   '     v2 值得跑，继续拿完整训练代码。')
    elif bot_sem_n >= 10:
        verdict = '⚠️  地基弱'
        advice  = ('bot-30 语义 token 偏少，c_a 信号有噪声。\n'
                   '     建议先做快速消融（单条 StoH run），再决定是否全跑。')
    else:
        verdict = '❌  地基垮'
        advice  = ('bot-30 被标点/子词/停用词占据，c_a = 噪声。\n'
                   '     v2 不值得跑，直接写②（诚实负结果版本）。')

    print(f'  结论: {verdict}')
    print(f'     {advice}')
    print(f'{"="*W}\n')


if __name__ == '__main__':
    main()
