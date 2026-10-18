"""Baseline: target model's own confidence (mean and sum token log-prob of its greedy answer), same prompt/settings as generate_answers.py."""
import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import *
from generate_answers import SYSTEM, PROMPT_TEMPLATE, GEN

def main():
    df = pd.read_csv("results/sample.csv")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=torch.float16).to(dev).eval()
    rows = []
    for i, q in enumerate(df.question):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
        with torch.no_grad():
            o = model.generate(**ids, **GEN, pad_token_id=tok.eos_token_id, return_dict_in_generate=True, output_scores=True)
        gen = o.sequences[0, ids["input_ids"].shape[1]:]
        lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(o.scores, gen)]
        text = tok.decode(gen, skip_special_tokens=True)
        # keep only tokens of the first line (the graded answer)
        keep, acc = [], ""
        for t, lp in zip(gen, lps):
            acc += tok.decode(t, skip_special_tokens=True)
            if "\\n" in acc.replace("\n", "\\n") and acc.strip() and "\n" in acc: break
            keep.append(lp)
        keep = keep or lps[:1]
        rows.append({"id": df.id[i], "answer_check": text.strip().split("\n")[0], "mean_logprob": sum(keep) / len(keep), "sum_logprob": sum(keep), "first_logprob": keep[0]})
        if i % 50 == 0: print(i, flush=True)
    pd.DataFrame(rows).to_csv("results/baseline_logprob.csv", index=False)

if __name__ == "__main__":
    main()
