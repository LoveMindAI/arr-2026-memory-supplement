"""Pod-side: generate memory predictions for val+test with the base model or a LoRA adapter (one GPU per process).
Usage: CUDA_VISIBLE_DEVICES=i python eval_memory_vllm.py --tag base|epoch1|... [--adapter DIR] --files f1.jsonl f2.jsonl
Same prompt (chat template, enable_thinking=False) as training; temperature 0.7, top_p 0.95, per-item seed; no truncation."""
import argparse, hashlib, json, os, time
from pathlib import Path
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
R = Path("/workspace/g31b")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--adapter"); ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--model", help="local model dir (e.g. a merged round-1 model); default = MANIFEST base")  # Oct 8; all else unchanged
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    m = json.loads((R / "MANIFEST.json").read_text())
    llm = LLM(**(dict(model=a.model) if a.model else dict(model=m["model"], revision=m["revision"])), dtype="bfloat16", max_model_len=8192, max_num_seqs=64, gpu_memory_utilization=0.90,
              enable_prefix_caching=False, seed=101, enable_lora=bool(a.adapter), max_lora_rank=16, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    lora = LoRARequest(a.tag, 1, a.adapter) if a.adapter else None
    rows = [json.loads(l) for f in a.files for l in open(f) if l.strip()]
    ps = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    mx = llm.llm_engine.vllm_config.model_config.max_model_len
    assert all(len(tok.encode(p, add_special_tokens=False)) + a.max_tokens <= mx for p in ps), "no truncation allowed"
    sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=a.max_tokens, seed=int(hashlib.sha256(r["perspectiveID"].encode()).hexdigest()[:7], 16)) for r in rows]
    t0 = time.time()
    res = llm.generate(ps, sp, lora_request=lora, use_tqdm=False)
    out = R / "eval_out"; out.mkdir(exist_ok=True)
    with open(out / f"{a.tag}.jsonl", "w") as f:
        for r, x in zip(rows, res):
            z = x.outputs[0]
            f.write(json.dumps(dict(perspectiveID=r["perspectiveID"], split=r["split"], tag=a.tag, adapter=a.adapter, text=z.text,
                                    finish=z.finish_reason, prompt_tokens=len(x.prompt_token_ids), output_tokens=len(z.token_ids))) + "\n")
    (out / f"{a.tag}.DONE").write_text(json.dumps(dict(n=len(rows), seconds=time.time() - t0)))


if __name__ == "__main__":
    main()
