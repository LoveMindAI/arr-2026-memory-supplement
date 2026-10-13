"""One-trip GPU job (Oct 7). Writers: lcw150 (LoRA r96) and plain Gemma 4 31B-it. Kit card sampling (T .7, top-p .95, max 4096), seed per pid.
A) weave v3 over the existing narrative cards, all people, both writers      -> weave_out/weave_all.jsonl
B) fact narrative ("who not how" v1), all people, both writers               -> weave_out/factnarr.jsonl
C) narrative cards from impression facets (kit card prompt), lcw150 only      -> weave_out/cards_impression.jsonl"""
import hashlib, json, os, time
from pathlib import Path
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
W = Path("/workspace"); IN = W / "weave_in"; OUT = W / "weave_out"; OUT.mkdir(exist_ok=True)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
m = json.loads((W / "g31b" / "MANIFEST.json").read_text())
llm = LLM(model=m["model"], revision=m["revision"], dtype="bfloat16", enable_lora=True, max_lora_rank=128, max_loras=1, max_model_len=16384,
          max_num_seqs=96, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
tok = llm.get_tokenizer()
WR = LoRARequest("writer_lcw150", 1, str(W / "adapters" / "writer_lcw150"))
render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
seed_of = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:7], 16)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
bullets = lambda xs: "\n".join(f"- {x}" for x in xs)


def run(name, jobs, prompts, writers, path):
    sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096, seed=seed_of(j["pid"])) for j in jobs]
    with open(path, "a") as f:
        for wname, lora in writers:
            t0 = time.time(); res = llm.generate([render(p) for p in prompts], sp, lora_request=lora, use_tqdm=False)
            for j, r in zip(jobs, res):
                z = r.outputs[0]
                f.write(json.dumps(dict(pid=j["pid"], writer=wname, text=z.text, finish=z.finish_reason, out_tokens=len(z.token_ids)), ensure_ascii=False) + "\n")
            log(name, wname, len(jobs), f"{time.time()-t0:.0f}s")


BOTH = (("lcw150", WR), ("gemma4_31b_base", None))
J = [json.loads(l) for l in open(IN / "weave_all_jobs.jsonl")]; T = (IN / "WEAVE_PROMPT_v3.md").read_text()
run("weave", J, [[{"role": "user", "content": T.format(traits=j["traits"], card=j["card"], facts=bullets(j["facts"]))}] for j in J], BOTH, OUT / "weave_all.jsonl")
J = [json.loads(l) for l in open(IN / "factnarr_jobs.jsonl")]; T = (IN / "FACT_NARRATIVE_PROMPT_v1.md").read_text()
run("factnarr", J, [[{"role": "user", "content": T.format(facts=bullets(j["facts"]))}] for j in J], BOTH, OUT / "factnarr.jsonl")
J = [json.loads(l) for l in open(IN / "impression_card_prompts.jsonl")]
run("impression_cards", J, [j["messages"] for j in J], (("lcw150", WR),), OUT / "cards_impression.jsonl")
(OUT / "WEAVE_DONE").write_text("ok")
