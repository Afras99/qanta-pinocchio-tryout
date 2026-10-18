# Pinocchio on QANTA: does its confidence help skip wrong answers?

## Summary
I asked a small chatbot (Qwen2.5-3B) 500 quiz-bowl questions from QANTA, and it got **65% right**. Pinocchio then looked at each question and answer and gave a confidence score, without ever seeing the correct answer.

- **Its confidence is useful.** Answers with higher scores really were right more often. Answering only the 40% of questions Pinocchio was most confident about raised accuracy from 65% to **84.5%**.
- **Its percentages run low.** On average it said 42% while 65% of answers were right. A quick fix using 100 labelled answers (Platt scaling) mostly corrected this, and did not change which answers it prefers.
- **What this tests:** Pinocchio as a confidence checker for another model's answers, not how well the chatbot knows what it knows.

## Setup
- **Data:** `qanta.dev.2021.12.20.json`; 500 random questions (seed 0), full question text.
- **Chatbot:** `Qwen/Qwen2.5-3B-Instruct`, fixed settings, short answers. System message: "You are an expert quiz bowl player." User message: "Answer the following quiz bowl question. Respond with ONLY the short answer (an entity or title, a few words at most), no explanation." followed by "Question: {question}" and "Answer:".
- **Marking answers right or wrong:** text matching against the official answer and its listed alternatives (ignoring case, punctuation, accents and plurals; allowing small spelling differences and a shared whole-word part of at least 5 characters). 65.0% right.
- **Pinocchio:** `pinocchio-uq` 0.1.1, told which chatbot wrote each answer.
- **Code:** https://github.com/Afras99/qanta-pinocchio-tryout

## Results

**1. Does Pinocchio's confidence separate right answers from wrong ones? Yes, fairly well.**
Right answers got an average confidence of 0.51, wrong answers 0.27. Splitting at 0.5: the 200 answers scored 0.5 or higher were right **84.5%** of the time, and the other 300 only **52%**. As one number, AUROC is **0.75** (95% interval 0.70 to 0.79): pick one right and one wrong answer at random, and the right one has the higher score about 75% of the time (a coin flip would be 50%).

**2. So what if we skip the doubtful questions?**
Instead of answering all 500 questions, answer only the ones Pinocchio is most confident about. Accuracy goes up as we answer fewer:

![Answering fewer questions gets more of them right](figures/abstention_curve.png)

*Figure 1. Moving left, we answer only the questions Pinocchio trusts most, and a larger share of those answers are right. Skipping questions at random (orange) does nothing.*

| Questions we answer | Right (Pinocchio picks them) | Right (picked at random) |
|---|---|---|
| All 500 (100%) | 65.0% | 65.0% |
| 400 (80%) | 72.8% | 65.0% |
| 300 (60%) | 80.7% | 65.0% |
| 200 (40%) | 84.5% | 64.9% |

The price is answering fewer questions.

**3. Are its percentages honest? No, they run too low.**
Pinocchio's confidence averaged 42%, but 65% of the answers were actually right. Its claimed percentages are off from the real hit rate by about 23 points on average (this is the "ECE", 0.23; 0 would be perfectly honest).

## Fixing the percentages (Platt scaling)
We can correct this with a small adjustment (Platt scaling): learn from 100 answers where we know the truth how to nudge Pinocchio's numbers, then test on the other 400. The gap shrinks from 0.236 to **0.064**. The ranking is untouched (AUROC 0.75 before and after), because the adjustment never reorders answers, so which questions get skipped stays the same.

![Before and after Platt scaling](figures/reliability_platt.png)

*Figure 2. Answers are grouped by Pinocchio's confidence (left to right); the height is how often each group was really right; the dashed line means perfectly honest (numbers = answers in the group). Before, nearly every group sits above the line, so Pinocchio was too low. After, the upper range lines up, but the lowest-score groups are still off (46 answers near 0.35 are right only 26% of the time).*

## Things to know
- **The right/wrong marking is imperfect.** On a fresh check of 100 answers, 7 were marked wrongly (4 too generous, 3 too strict), so accuracy is slightly overstated. The checks were done by me with AI help and re-read by a second AI reviewer; no independent human checked them.
- **Telling Pinocchio the chatbot's name** does not change how well it ranks answers (AUROC 0.75 either way) but changes how low its percentages run (ECE 0.23 with the name, 0.09 without).
- **A simpler signal ranked answers better:** the chatbot's own internal confidence (AUROC 0.84), though it needs access to the chatbot's internals, which Pinocchio does not.
- **Limits:** one chatbot, one 500-question sample, dev set only.
