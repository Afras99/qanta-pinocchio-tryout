"""Generate short answers with a local target model (greedy decoding)."""
import json, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import *

SYSTEM = "You are an expert quiz bowl player."
PROMPT_TEMPLATE = (
    "Answer the following quiz bowl question. Respond with ONLY the short answer "
    "(an entity or title, a few words at most), no explanation.\n\n"
    "Question: {question}\n\nAnswer:"
)
GEN = dict(do_sample=False, max_new_tokens=24)

def main():
    df = pd.read_csv("results/sample.csv")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=torch.float16).to(dev).eval()
    outs = []
    for i, q in enumerate(df.question):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT_TEMPLATE.format(question=q)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
        with torch.no_grad():
            o = model.generate(**ids, **GEN, pad_token_id=tok.eos_token_id)
        a = tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
        outs.append(a)
        if i % 25 == 0: print(i, repr(a), flush=True)
    df["gold"] = df["page"]
    df["model_answer"] = outs
    df[["id", "question", "gold", "answer", "model_answer"]].to_csv("results/answers.csv", index=False)
    json.dump({"model": TARGET_MODEL, "system": SYSTEM, "prompt_template": PROMPT_TEMPLATE, "generation": GEN, "dtype": "float16", "device": dev},
              open("results/generation_config.json", "w"), indent=2)

if __name__ == "__main__":
    main()
