"""Smoke test: score hand-written pairs; save results to results/smoke_test.json."""
import json, time, torch
from pinocchio import Pinocchio
t = time.time(); judge = Pinocchio(); load_s = time.time() - t
out = {"load_seconds": round(load_s, 1), "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "pairs": []}
for q, a in [("What is the capital of Japan?", "Tokyo"), ("What is the capital of Japan?", "Kyoto")]:
    t = time.time(); s = judge.score(question=q, answer=a)
    out["pairs"].append({"question": q, "answer": a, "p_correct": s, "seconds": round(time.time() - t, 2)})
out["tokyo_gt_kyoto"] = out["pairs"][0]["p_correct"] > out["pairs"][1]["p_correct"]
json.dump(out, open("results/smoke_test.json", "w"), indent=2); print(json.dumps(out, indent=2))
