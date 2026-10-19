# Pinocchio confidence scores on QANTA: does abstention help?

Tryout project: test whether [Pinocchio](https://github.com/khayes95/pinocchio) (a small external model that scores how likely another model's answer is to be correct) is a useful confidence signal on QANTA quiz-bowl questions.

**Result in one line:** on 500 QANTA dev questions answered by Qwen2.5-3B-Instruct (65.0% correct), answering only the 40% of questions Pinocchio trusts most gives 84.5% accuracy; its raw probabilities are under-confident (ECE 0.23), and Platt scaling on 100 labelled examples reduces ECE to 0.06 without changing the ranking. With only the first sentence of each question the chatbot is right 7.6% of the time and Pinocchio's AUROC falls to 0.56 (0.75 on the full question). See [`WRITEUP.md`](WRITEUP.md) for the full report.

## Layout
```
src/        pipeline code (one script per stage)
results/    raw outputs: answers, grades, scores, metrics (CSV/JSON)
figures/    plots used in the writeup
WRITEUP.md  the report (WRITEUP.pdf is the same report, 2 pages)
```
`data/` (the QANTA dev file) is not committed; the command below downloads it.

## Reproduce
```
uv venv --python 3.13 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
mkdir -p data && curl -L -o data/qanta.dev.2021.12.20.json https://huggingface.co/datasets/TasnimKabir12/qanta/resolve/main/qanta.dev.2021.12.20.json
export PYTHONPATH=src                       # run everything from the repo root
python src/smoke_test.py                    # sanity check -> results/smoke_test.json
python src/load_data.py                     # sample 500 questions (seed 0)
python src/generate_answers.py              # Qwen2.5-3B-Instruct answers (Apple GPU/MPS)
python src/grade.py                         # labels -> results/graded.csv (labels live only here)
python src/audit.py                         # verify the saved 100-example grader audit sample
python src/score.py --source-model          # Pinocchio scores (primary; ~24 min on CPU)
python src/score.py                         # same without source_model (comparison)
python src/evaluate.py                      # AUROC, ECE, abstention table and figures
python src/recalibrate.py                   # Platt scaling extension
python src/baseline_logprob.py              # target model's own token-probability baseline
python src/control_shuffled.py              # shuffled-answer control (200 pairs, ~10 min)
python src/compare_source.py                # with vs without source_model
python src/analysis_extra.py                # robustness checks -> results/robustness.json
python src/prefix_data.py                   # first sentence / first two / first half / full versions of the 500 questions (cuts snapped to true sentence ends)
python src/prefix_generate.py               # answers for each prefix (~35 min on Apple GPU; incremental: reruns only changed prefixes)
python src/prefix_grade.py                  # same grader; asserts the full-question labels equal results/graded.csv
python src/prefix_score.py                  # Pinocchio scores for the prefixes (~65 min on CPU; incremental)
python src/prefix_evaluate.py               # per-level metrics, figure, seeded-random examples -> results/prefix_*.json
python src/build_pdf.py                     # WRITEUP.md -> WRITEUP.pdf (needs Google Chrome)
```
Seeds are fixed (0) and dependencies are pinned in `requirements.txt`. Generation runs in fp16 on an Apple GPU (MPS), so answers may not be bit-identical on other hardware.
