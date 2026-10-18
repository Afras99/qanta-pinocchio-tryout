import re, unicodedata
SEED = 0
N_SAMPLE = 500
DATA = "data/qanta.dev.2021.12.20.json"
TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct"

def normalize(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.replace("_", " ")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return " ".join(s.split())

PRIMARY_SCORES = "results/scores_source_model.csv"   # scored with source_model=TARGET_MODEL (passed because the answering model is known)
NO_SOURCE_SCORES = "results/scores.csv"              # same pairs, source_model left empty (comparison)

def load_scored(path=PRIMARY_SCORES):
    """Raw Pinocchio scores joined with the current grading labels (labels live only in graded.csv)."""
    import pandas as pd
    s = pd.read_csv(path)[["id", "p_correct"]]
    return s.merge(pd.read_csv("results/graded.csv")[["id", "correct", "correct_strict"]], on="id", validate="1:1")
