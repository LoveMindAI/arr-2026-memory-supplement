"""DPO for the story writer (Sep 30, agent1). Same base, revision, LoRA recipe (r16/a32, dropout .05, same target-module rule), AdamW wd .01,
effective batch 8 (pairs), 5% warmup + linear decay, grad clip 1.0 as gpu_worker_ddp.py. Differences, all deliberate:
 - policy start AND reference = M03 merged into the base weights ('merge' mode writes /workspace/m03_merged once; 'train' loads it).
   The new LoRA starts at zero, so policy == reference at step 0 and the first DPO loss must be ln2 = 0.6931 (asserted: a built-in correctness check).
 - reference log-probs = the same model with the new adapter disabled, precomputed once before training (REF_LOGPS.json).
 - loss = -logsigmoid(beta * ((logp_c - ref_c) - (logp_r - ref_r))), summed completion-token log-probs (standard DPO).
 - no DDP: each rank runs its share of the 8 pairs, then gradients are all-reduced (averaged) by hand before clipping/stepping
   (two sequences per pair; avoids DDP's one-forward-per-backward constraint). Initial LoRA weights are broadcast from rank 0.
Round 3 additions (Oct 2, agent1; all default OFF, so old runs reproduce exactly):
 --lr-floor F       linear decay ends at F x peak instead of 0 (decay-to-0 squeezed the reader's spread in run A).
 --nll-alpha A      rows with "nll": true also get A x (mean per-token NLL of the chosen completion) (RPO-style; helps learn off-policy chosen texts).
 --rehearsal FILE --rehearsal-per-step K --rehearsal-weight W
                    K reader-training rows ({messages, completion}) per optimizer step, loss W x mean per-token NLL of the completion (protects reading).
 The step-1 ln2 check uses the DPO term only.
 --len-norm (Oct 5)  use MEAN per-token log-probs of chosen/rejected (policy and reference) instead of sums, so longer completions get no free margin
                    (the Oct 4 both-better run learned 'longer is better'); needs a larger beta (per-token log-ratios are ~1/n of summed ones).
Usage: torchrun ... dpo_worker.py merge --m03 DIR
       torchrun ... dpo_worker.py train --tag NAME --pairs FILE --lr X --beta B [--limit N] [--steps-cap K] [--lr-floor F] [--nll-alpha A] [--rehearsal FILE ...]"""
import os, sys, json, time, math, random, argparse, datetime, contextlib, hashlib, shutil, glob, traceback
from pathlib import Path
import gpu_worker as G
MERGED = Path('/workspace/m03_merged')   # default; --out (merge) / --base (train) choose another, e.g. /workspace/w1_merged

def main():
    import torch, transformers, peft
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['merge', 'train']); ap.add_argument('--m03'); ap.add_argument('--out', default=str(MERGED)); ap.add_argument('--base', default=str(MERGED)); ap.add_argument('--tag'); ap.add_argument('--pairs')
    ap.add_argument('--lr', type=float); ap.add_argument('--beta', type=float, default=0.1); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--steps-cap', type=int, default=0)
    ap.add_argument('--lr-floor', type=float, default=0.); ap.add_argument('--nll-alpha', type=float, default=0.); ap.add_argument('--rehearsal')
    ap.add_argument('--rehearsal-per-step', type=int, default=0); ap.add_argument('--rehearsal-weight', type=float, default=1.); ap.add_argument('--len-norm', action='store_true'); a = ap.parse_args()
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1)); local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local); device = torch.device('cuda', local); torch.set_num_threads(4)
    if world > 1: dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=60), device_id=device)
    m = json.loads((G.R / 'MANIFEST.json').read_text())
    tok = AutoTokenizer.from_pretrained(m['model'], revision=m['revision'])
    fixture = [{'role': 'user', 'content': 'Write one sentence about a quiet library.'}]
    fi = tok(G.prompt(tok, fixture), return_tensors='pt', add_special_tokens=False).to(device)
    if a.mode == 'merge':
        OUTP = Path(a.out)
        assert world == 1
        model = AutoModelForImageTextToText.from_pretrained(m['model'], revision=m['revision'], dtype=torch.bfloat16, device_map={'': local}, attn_implementation='sdpa')
        with torch.no_grad(): base_l = model(**fi, use_cache=False, logits_to_keep=1).logits.float().clone()
        c = json.load(open(Path(a.m03) / 'COMPLETE.json')); files = c.get('files', c)
        assert hashlib.sha256(open(Path(a.m03) / 'adapter_model.safetensors', 'rb').read()).hexdigest() == files['adapter_model.safetensors']
        pm = PeftModel.from_pretrained(model, a.m03); pm.eval()
        with torch.no_grad(): lora_l = pm(**fi, use_cache=False, logits_to_keep=1).logits.float().clone()
        merged = pm.merge_and_unload(); merged.eval()
        with torch.no_grad(): merged_l = merged(**fi, use_cache=False, logits_to_keep=1).logits.float()
        info = {'m03': a.m03, 'm03_sha256': files['adapter_model.safetensors'], 'adapter_shift_vs_base_max': float((lora_l - base_l).abs().max()),
                'merged_vs_unmerged_max': float((merged_l - lora_l).abs().max()), 'merged_vs_base_max': float((merged_l - base_l).abs().max())}
        assert info['adapter_shift_vs_base_max'] > 1e-3 and info['merged_vs_unmerged_max'] < 0.25 * info['adapter_shift_vs_base_max'], info
        tmp = Path(str(OUTP) + '.tmp'); shutil.rmtree(tmp, ignore_errors=True)
        merged.save_pretrained(tmp, safe_serialization=True, max_shard_size='5GB'); tok.save_pretrained(tmp)
        snap = Path(glob.glob(f"{os.environ.get('HF_HOME', '/workspace/hf')}/hub/models--{m['model'].replace('/', '--')}/snapshots/{m['revision']}")[0])
        for f in snap.iterdir():   # processor / chat template / generation config etc. that save_pretrained may not write
            if f.is_file() and f.suffix != '.safetensors' and not f.name.endswith('.index.json') and not (tmp / f.name).exists(): shutil.copy(f, tmp / f.name)
        G.save(tmp / 'MERGE_INFO.json', info); tmp.rename(OUTP); print('MERGED', json.dumps(info), flush=True); return
    # ---------------- train
    td = G.OUT / 'training' / a.tag
    BASE = Path(a.base)
    model = AutoModelForImageTextToText.from_pretrained(str(BASE), dtype=torch.bfloat16, device_map={'': local}, attn_implementation='sdpa')
    if (BASE / 'tokenizer.json').exists(): tok = AutoTokenizer.from_pretrained(str(BASE))   # Oct 7: the base's own tokenizer (26B)
    if (BASE / 'tokenizer.json').exists(): fi = tok(G.prompt(tok, fixture), return_tensors='pt', add_special_tokens=False).to(device)   # Oct 7: re-encode the fixture with that tokenizer
    PROJ = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    modules = [n for n, v in model.named_modules() if isinstance(v, torch.nn.Linear) and 'language_model' in n and n.rsplit('.', 1)[-1] in PROJ]
    _tc = getattr(model.config, 'text_config', model.config)
    assert len(modules) >= 5 * _tc.num_hidden_layers and all('vision' not in n for n in modules)   # Oct 7: was >= 300 (31B); 26B-A4B has 30 layers
    model.eval()
    with torch.no_grad(): original = model(**fi, use_cache=False, logits_to_keep=1).logits.detach().clone()
    torch.manual_seed(17); random.seed(17)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, target_modules=modules, task_type=TaskType.CAUSAL_LM, bias='none'))
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        for p in params: dist.broadcast(p.data, src=0)
    model.eval()
    with torch.no_grad(): delta = float((original - model(**fi, use_cache=False, logits_to_keep=1).logits).abs().max())
    assert delta < 1e-5, delta; del original
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.enable_input_require_grads(); model.config.use_cache = False
    def enc(msgs, completion):
        p = G.prompt(tok, msgs); text = tok.apply_chat_template(msgs + [{'role': 'assistant', 'content': completion}], tokenize=False, add_generation_prompt=False, enable_thinking=False)
        empty = '<|channel>thought\n<channel|>'
        if not text.startswith(p): assert p.endswith(empty); b = p[:-len(empty)]; assert text.startswith(b); text = p + text[len(b):]
        ids = tok.encode(text, add_special_tokens=False); pre = tok.encode(p, add_special_tokens=False); assert ids[:len(pre)] == pre and len(ids) <= 16384
        return ids, len(pre)
    rows = G.rows(Path(a.pairs))
    if a.limit: rows = rows[:a.limit]
    data = []
    for r in rows:
        ic, pc = enc(r['messages'], r['chosen']); ir, pr = enc(r['messages'], r['rejected']); assert pc == pr and ic[:pc] == ir[:pr]
        data.append({'key': r['person_key'] + ('|style' if r.get('nll') else ''), 'c': ic, 'r': ir, 'P': pc, 'nll': bool(r.get('nll'))})
    assert len({d['key'] for d in data}) == len(data), 'duplicate pair keys'
    reh = []
    if a.rehearsal and a.rehearsal_per_step:
        for r in G.rows(Path(a.rehearsal)):
            ids, P = enc(r['messages'], r['completion']); reh.append({'x': ids, 'P': P})
        random.Random(19).shuffle(reh)
    N = len(data); N8 = (N // 8) * 8; assert N8 >= 8
    PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
    def logps(net, d):   # summed completion log-probs for chosen and rejected, one padded batch of 2
        L = max(len(d['c']), len(d['r'])); P = d['P']
        x = torch.tensor([s + [PAD] * (L - len(s)) for s in (d['c'], d['r'])], device=device)
        am = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in (d['c'], d['r'])], device=device)
        idx = torch.arange(P - 1, L - 1, device=device); y = x[:, P:]
        mask = torch.tensor([[1.] * (len(s) - P) + [0.] * (L - len(s)) for s in (d['c'], d['r'])], device=device)
        logits = net(input_ids=x, attention_mask=am, use_cache=False, logits_to_keep=idx).logits
        assert logits.shape[1] == y.shape[1]
        def chunk(lg, yy, mm): return (torch.gather(F.log_softmax(lg.float(), -1), -1, yy.unsqueeze(-1)).squeeze(-1) * mm).sum(1)
        tot = sum(checkpoint(chunk, logits[:, i:i + 256, :], y[:, i:i + 256], mask[:, i:i + 256], use_reentrant=False) for i in range(0, y.shape[1], 256))
        return tot[0], tot[1]
    def sft_nll(net, d):   # mean per-token NLL of one completion (rehearsal rows)
        x = torch.tensor([d['x']], device=device); P = d['P']; L = len(d['x']); idx = torch.arange(P - 1, L - 1, device=device); y = x[:, P:]
        lg = net(input_ids=x, use_cache=False, logits_to_keep=idx).logits; assert lg.shape[1] == y.shape[1]
        return -torch.gather(F.log_softmax(lg.float(), -1), -1, y.unsqueeze(-1)).squeeze(-1).mean()
    # reference log-probs (adapter disabled = merged M03), precomputed once
    model.eval(); ref = {}
    with torch.no_grad(), model.disable_adapter():
        for j in range(rank, N, world):
            c, r_ = logps(model, data[j]); ref[j] = (float(c), float(r_))
    if world > 1:
        allr = [None] * world; dist.all_gather_object(allr, ref); ref = {k: v for d in allr for k, v in d.items()}
    assert len(ref) == N
    if rank == 0: G.save(td / 'REF_LOGPS.json', {'pairs': N, 'ref': {data[j]['key']: ref[j] for j in range(N)}})
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=.01)
    total = N8 // 8 if not a.steps_cap else min(N8 // 8, a.steps_cap); acc = 8 // world; assert 8 % world == 0
    if rank == 0: G.save(td / 'RECIPE.json', {'method': 'DPO', 'start_and_reference': str(BASE) + ' (adapter ' + (lambda mi: mi.get('m03_sha256') or mi.get('adapter_sha256'))(json.load(open(BASE / 'MERGE_INFO.json')))[:12] + ' merged into base)',
        'beta': a.beta, 'lr': a.lr, 'steps': total, 'pairs': N, 'rank': 16, 'alpha': 32, 'dropout': .05, 'world_size': world, 'effective_batch_pairs': 8, 'schedule': f'5% warmup, linear decay to {a.lr_floor} x peak',
        'len_norm': a.len_norm, 'nll_alpha': a.nll_alpha, 'nll_pairs': sum(d['nll'] for d in data), 'rehearsal': a.rehearsal, 'rehearsal_rows': len(reh), 'rehearsal_per_step': a.rehearsal_per_step, 'rehearsal_weight': a.rehearsal_weight,
        'pairs_sha256': G.sha(Path(a.pairs)), 'worker_sha256': G.sha(Path(__file__)), 'zero_adapter_max_logit_error': delta, 'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}})
    order = list(range(N)); random.Random(18).shuffle(order); torch.manual_seed(17 + rank); model.train(); start = time.time()
    for step in range(1, total + 1):
        batch = order[(step - 1) * 8: step * 8]; opt.zero_grad(set_to_none=True); st = {'loss': 0., 'acc': 0., 'margin': 0., 'dc': 0., 'dr': 0., 'nll': 0., 'n_nll': 0., 'reh': 0.}; t0 = time.time()
        for j in batch[rank::world]:
            c, r_ = logps(model, data[j]); rc, rr = ref[j]
            if a.len_norm:
                nc, nr = len(data[j]['c']) - data[j]['P'], len(data[j]['r']) - data[j]['P']
                z = a.beta * ((c - rc) / nc - (r_ - rr) / nr)
            else: z = a.beta * ((c - rc) - (r_ - rr))
            loss = -F.logsigmoid(z); tot = loss
            if a.nll_alpha and data[j]['nll']:
                nl = -c / (len(data[j]['c']) - data[j]['P']); tot = tot + a.nll_alpha * nl; st['nll'] += float(nl); st['n_nll'] += 1
            (tot / 8).backward()   # mean over the 8 pairs of the global batch (grads summed across ranks below)
            st['loss'] += float(loss) / 8; st['acc'] += float(z > 0) / 8; st['margin'] += float(z) / 8; st['dc'] += (float(c) - rc) / 8; st['dr'] += (float(r_) - rr) / 8
        if reh:
            K = a.rehearsal_per_step
            for q in range(rank, K, world):
                d = reh[((step - 1) * K + q) % len(reh)]; nl = sft_nll(model, d); (a.rehearsal_weight * nl / K).backward(); st['reh'] += float(nl) / K
        if world > 1:
            for p in params:
                if p.grad is None: p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
            v = torch.tensor([st[k] for k in st], device=device, dtype=torch.float64); dist.all_reduce(v); st = dict(zip(st, v.tolist()))
        if step == 1: assert abs(st['loss'] - math.log(2)) < 2e-3, ('policy != reference at step 0', st)   # the built-in check
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.)); assert math.isfinite(norm)
        warm = max(1, round(total * .05)); scale = (a.lr_floor + (1. - a.lr_floor) * max(0., (total - step) / (total - warm))) if step > warm else step / warm
        for g in opt.param_groups: g['lr'] = a.lr * scale
        opt.step(); torch.cuda.synchronize()
        log = {'time': time.time(), 'step': step, **st, 'grad_norm': norm, 'lr': a.lr * scale, 'elapsed': time.time() - start, 'step_seconds': time.time() - t0, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9}
        if rank == 0:
            td.mkdir(parents=True, exist_ok=True)
            with (td / 'steps.jsonl').open('a') as f: f.write(json.dumps(log) + '\n')
    if rank == 0:
        cp = td / 'epoch1'; model.save_pretrained(cp); tok.save_pretrained(cp)
        G.save(cp / 'COMPLETE.json', {'step': total, 'time': time.time(), 'files': {p.name: G.sha(p) for p in cp.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}})
    if world > 1: dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    try: main()
    except Exception:
        traceback.print_exc(); raise
