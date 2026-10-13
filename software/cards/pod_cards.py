"""Pod-side card pipeline on ONE GPU: lcw150 writes narrative cards -> lcw150 person-specific pass -> reader B reads all cards.
Base google/gemma-4-31B-it @842da379. Adapters: /workspace/adapters/writer_lcw150 (r96), /workspace/adapters/reader_gemma_rvB (r16).
Chat template: add_generation_prompt=True, enable_thinking=False. Writer sampling = kit card sampling (T .7, top-p .95, max 4096, seed 101+);
reader = greedy, max 512 (as reader_eval_worker.py). Outputs in /workspace/cards_out/."""
import hashlib, json, os, sys, time
from pathlib import Path
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
W = Path("/workspace"); IN = W / "cards_in"; OUT = W / "cards_out"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(W / "kit"))
import prompts as KIT
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

m = json.loads((W / "g31b" / "MANIFEST.json").read_text())
llm = LLM(model=m["model"], revision=m["revision"], dtype="bfloat16", enable_lora=True, max_lora_rank=128, max_loras=2, max_model_len=16384,
          max_num_seqs=96, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
tok = llm.get_tokenizer()
WR = LoRARequest("writer_lcw150", 1, str(W / "adapters" / "writer_lcw150"))
RD = LoRARequest("reader_B", 2, str(W / "adapters" / "reader_gemma_rvB"))
render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
seed_of = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:7], 16)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# 1) narrative cards
cards = {}
p1 = OUT / "cards_lcw150.jsonl"
if p1.exists():
    cards = {json.loads(l)["pid"]: json.loads(l)["text"] for l in open(p1)}
else:
    jobs = [json.loads(l) for l in open(IN / "card_prompts.jsonl")]
    t0 = time.time()
    res = llm.generate([render(j["messages"]) for j in jobs],
                       [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096, seed=101) for j in jobs], lora_request=WR, use_tqdm=False)
    with open(p1, "w") as f:
        for j, r in zip(jobs, res):
            z = r.outputs[0]; cards[j["pid"]] = z.text
            f.write(json.dumps(dict(pid=j["pid"], writer="lcw150", text=z.text, finish=z.finish_reason, out_tokens=len(z.token_ids))) + "\n")
    log("narrative cards", len(cards), f"{time.time()-t0:.0f}s")

# 2) person-specific pass (leave-one-call-out facts + trait reference)
TPL = (IN / "PERSON_PASS_PROMPT_v2.md").read_text()
p2 = OUT / "cards_person.jsonl"
if not p2.exists():
    jobs = [json.loads(l) for l in open(IN / "person_jobs.jsonl")]
    todo = [j for j in jobs if j["facts"]]
    msgs = [[{"role": "user", "content": TPL.format(traits=j["traits"], card=cards[j["pid"]], facts="\n".join(f"- {x}" for x in j["facts"]))}] for j in todo]
    t0 = time.time()
    res = llm.generate([render(x) for x in msgs], [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096, seed=seed_of(j["perspectiveID"])) for j in todo],
                       lora_request=WR, use_tqdm=False)
    with open(p2, "w") as f:
        for j, r in zip(todo, res):
            z = r.outputs[0]
            f.write(json.dumps(dict(perspectiveID=j["perspectiveID"], pid=j["pid"], writer="lcw150_personpass", n_facts=len(j["facts"]), text=z.text,
                                    finish=z.finish_reason, out_tokens=len(z.token_ids))) + "\n")
        for j in jobs:
            if not j["facts"]:
                f.write(json.dumps(dict(perspectiveID=j["perspectiveID"], pid=j["pid"], writer="lcw150_narrative_nofacts", n_facts=0, text=cards[j["pid"]], finish="copy", out_tokens=0)) + "\n")
    log("person cards", len(jobs), f"{time.time()-t0:.0f}s")

# 3) reader B reads every card
items = [dict(kind="lcw150", key=k, text=v) for k, v in cards.items()]
items += [dict(kind=json.loads(l)["writer"], key=json.loads(l)["perspectiveID"], text=json.loads(l)["text"]) for l in open(p2) if json.loads(l)["writer"] == "lcw150_personpass"]
for l in open(IN / "frontier_cards.jsonl"):
    d = json.loads(l); items.append(dict(kind=d["writer"], key=d["pid"], text=d["text"]))
t0 = time.time()
rendered = [render(KIT.reader_messages("hexaco", it["text"])[0]) for it in items]
res = llm.generate(rendered, SamplingParams(temperature=0.0, max_tokens=512), lora_request=RD, use_tqdm=False)
with open(OUT / "reads_readerB.jsonl", "w") as f:
    for it, r in zip(items, res):
        f.write(json.dumps(dict(kind=it["kind"], key=it["key"], read=r.outputs[0].text, finish=r.outputs[0].finish_reason)) + "\n")
log("reads", len(items), f"{time.time()-t0:.0f}s")
(OUT / "CARDS_DONE").write_text("ok")
