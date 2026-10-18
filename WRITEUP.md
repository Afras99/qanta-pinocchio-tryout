# Does Pinocchio's confidence help a chatbot skip wrong answers? A test on QANTA

## 1. Summary
I had Qwen2.5-3B-Instruct answer 500 QANTA quiz-bowl questions (65.0% correct) and used Pinocchio, a small external model that scores how likely an answer is to be right, to rate each one. Its scores rank answers well (AUROC 0.75, 95% interval 0.70 to 0.79; 0.5 is chance), so skipping the answers it trusts least works as is: answering only the 40% of questions it trusts most lifts accuracy from 65.0% to 84.5%. Its raw probabilities are too low (average 42% vs 65% actually right; ECE 0.23, "off by about 23 points"), and Platt scaling fitted on 100 labelled answers cuts ECE to 0.06 on 400 unseen answers without changing the ranking. Caveats: how off the raw probabilities are depends on whether Pinocchio is told the chatbot's name (ECE 0.09 without it), and the chatbot's own token probabilities ranked answers better (AUROC 0.84) but need white-box access.

## 2. Setup
- **Data:** `qanta.dev.2021.12.20.json` (2,231 questions, none empty); 500 sampled at random (seed 0); full question text. **Code:** https://github.com/Afras99/qanta-pinocchio-tryout
- **Chatbot:** `Qwen/Qwen2.5-3B-Instruct`, fixed settings (no randomness), short answers. System message: "You are an expert quiz bowl player." User message: "Answer the following quiz bowl question. Respond with ONLY the short answer (an entity or title, a few words at most), no explanation." followed by "Question: {question}" and "Answer:".
- **Grading:** an answer is right if, after normalising (case, punctuation, accents, articles, plurals), it equals the Wikipedia page title, the official answer or a listed alternative; or is a near-spelling match (similarity ≥ 0.9, both strings ≥ 8 characters); or one is a whole-word part of the other (≥ 5 characters). **65.0% right** (52.2% with strict exact matching).
- **Pinocchio:** `pinocchio-uq` 0.1.1 (0.8B model), told which chatbot wrote each answer (the same pairs were also scored without that, section 5).
- **Hardware:** Apple M3 laptop, 16 GB; the chatbot ran on its GPU, Pinocchio on the CPU (about 24 minutes for 500 answers).

## 3. Results
| Measure (500 answers) | Value |
|---|---|
| AUROC: how often a right answer outscores a wrong one (0.5 = chance) | **0.75** (95% interval 0.70 to 0.79) |
| ECE: average gap between claimed and actual accuracy (0 = honest) | **0.23** (0.20 to 0.27); average score 0.42 vs 0.65 right |

![Answering fewer questions gets more of them right](figures/abstention_curve.png)

*Figure 1. Answering only the questions Pinocchio trusts most (left = fewer questions) raises the share that are right; skipping at random (orange) does nothing. Drawn from 5% coverage, below which too few answers remain.*

| Questions answered | Right (Pinocchio picks) | Right (random picks) | Gain (95% interval) |
|---|---|---|---|
| 100% | 65.0% | 65.0% | none |
| 80% | 72.8% | 65.0% | +7.8 points (5.4 to 9.6) |
| 60% | 80.7% | 65.0% | +15.7 points (12.0 to 18.7) |
| 40% | 84.5% | 64.9% | +19.5 points (14.4 to 24.1) |

## 4. Extension: Platt scaling
Platt scaling (Platt, 1999) is a two-number remapping, `new = sigmoid(0.568 × logit(old) + 0.843)`, learned from 100 labelled answers and tested on the other 400. It never reorders answers, so ranking is unchanged (AUROC 0.7505 before and after). ECE on the 400 unseen answers falls from **0.236 to 0.064**. Repeating with 200 different random 100/400 splits (resamples of the same 500 answers, so not independent) gives 0.232 to 0.074, improving every time.

![Before and after Platt scaling](figures/reliability_platt.png)

*Figure 2. Each dot is a group of answers: x is what Pinocchio said, y is how often they were right; the dashed line is perfectly honest (numbers = group sizes). Before, almost every group sits above the line (under-confident). After, the upper range lines up, but the lowest-score groups are still off: e.g. 46 answers near 0.35 are right only 26% of the time.*

## 5. Failures, checks and caveats
- **Grader errors.** On a fresh seeded audit of the final grader (100 answers, `src/audit.py`), 7 were wrong: 4 accepted too loosely (e.g. "Index" for "Refractive index"), 3 rejected too strictly (e.g. "Goya" for "Francisco Goya"), i.e. 7% (95% range 3% to 14%), so accuracy is slightly overstated. Labels were checked by the author with AI assistance, and a second AI reviewer re-read all 100; no independent human checked them. With strict exact-match labels AUROC is 0.69 and ECE 0.15, so absolute numbers depend on the grader but the ranking still beats chance.
- **Telling Pinocchio the chatbot's name** barely changes ranking (AUROC 0.748 vs 0.752) but changes calibration a lot: ECE 0.233 with it, 0.093 without. The ranking and abstention results do not depend on this choice; the raw calibration result does.
- **Pinocchio reads the answer:** pairing questions with answers from other questions drops its average score from 0.63 to 0.17 (200 pairs). Answer length, question length and "answer appears in the question" score near chance (AUROC 0.54, 0.41, 0.45).
- **A simpler signal ranks better:** the chatbot's own average token probability gives AUROC 0.84 (0.80 to 0.87; paired difference vs Pinocchio +0.09, interval 0.04 to 0.14) and 87.0% accuracy at 40% coverage, but needs access to the model's internals. One chatbot only.
- **Scope:** one chatbot, one 500-question sample, dev set only; training-data overlap not checked; results not split by difficulty or category; some page labels disagree with the official answers (59 of 500, either accepted); GPU generation may not be bit-reproducible. **Pinocchio scores another model's answers, so these results describe Pinocchio as a confidence estimator on QANTA, not the chatbot's own calibration.**

## 6. Next steps
1. Adapt Pinocchio's calibration/training recipe to QANTA data (here its outputs were only recalibrated, not retrained).
2. Evaluate on partial questions as clues are revealed, where early abstention matters most.
3. Replace text matching with an LLM judge or human labels, and repeat with several chatbots.
