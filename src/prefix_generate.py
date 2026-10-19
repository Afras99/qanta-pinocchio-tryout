"""Answer each prefix level with the same model, prompt and decoding settings as generate_answers.py.
Prefixes identical to the full question reuse the full-question answer; identical prefixes across levels are generated once.
Incremental: a prefix whose exact text already has an answer in results/prefix_answers.csv keeps it (greedy decoding), so rerunning after a
change to prefix_data.py regenerates only the prefixes whose text changed."""
import os, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import *
from generate_answers import SYSTEM, PROMPT_TEMPLATE, GEN

def main():
    pq = pd.read_csv("results/prefix_questions.csv")
    full = pd.read_csv("results/answers.csv").set_index("id").model_answer.to_dict()
    full_q = pq[pq.level == "full"].set_index("id").question.to_dict()
    cache = {}
    if os.path.exists("results/prefix_answers.csv"):
        for r in pd.read_csv("results/prefix_answers.csv").fillna("").itertuples(): cache[(r.id, r.question)] = r.model_answer
    for qid, text in full_q.items(): cache[(qid, text)] = full[qid]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=torch.float16).to(dev).eval()
    todo = pq[[(r.id, r.question) not in cache for r in pq.itertuples()]].drop_duplicates(["id", "question"])
    print("to generate:", len(todo), "of", len(pq), flush=True)
    for i, r in enumerate(todo.itertuples()):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT_TEMPLATE.format(question=r.question)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
        with torch.no_grad():
            o = model.generate(**ids, **GEN, pad_token_id=tok.eos_token_id)
        cache[(r.id, r.question)] = tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
        if i % 25 == 0: print(i, r.level, repr(cache[(r.id, r.question)]), flush=True)
    pq["model_answer"] = [cache[(r.id, r.question)] for r in pq.itertuples()]
    pq.to_csv("results/prefix_answers.csv", index=False)
    print("done")

if __name__ == "__main__":
    main()
