# Pinocchio on QANTA: does its confidence help skip wrong answers?

## Summary
I asked a small chatbot (Qwen2.5-3B) 500 quiz-bowl questions from QANTA, and it got **65% right**. Pinocchio then looked at each question and answer and gave a confidence score, without ever seeing the correct answer. I then repeated this with only the first part of each question, to see how much of the question Pinocchio needs.

- **On full questions its confidence is useful.** Answering only the 40% of questions Pinocchio was most confident about raised accuracy from 65% to **84.5%**.
- **It needs enough of the question.** After only the first sentence the chatbot is right just 8% of the time and Pinocchio can barely tell right from wrong (AUROC 0.56, close to a coin flip). Its ability grows with more text, to 0.75 on the full question.
- **Its percentages are off in both directions.** On full questions they run low (42% said, 65% right); on first sentences they run high (27% said, 8% right). A quick fix using 100 labelled answers (Platt scaling) mostly corrects the full-question case.
- **What this tests:** Pinocchio as a confidence checker for another model's answers, not how well the chatbot knows what it knows.

## Setup
- **Data:** `qanta.dev.2021.12.20.json`; 500 random questions (seed 0). **Chatbot:** `Qwen/Qwen2.5-3B-Instruct`, fixed settings, short answers (prompt in `src/generate_answers.py`). **Pinocchio:** `pinocchio-uq` 0.1.1, told which chatbot wrote each answer. **Code:** https://github.com/Afras99/qanta-pinocchio-tryout
- **How much of the question is seen:** the question cut at QANTA's own sentence boundaries (4 to 11 sentences, 6.5 on average): the **first sentence** (17% of the text on average), the **first two** (34%), the **first half** (56%) and the **full question**. The chatbot and Pinocchio both saw only the cut text; settings were otherwise identical. QANTA's character offsets drift by a few characters on 45 questions with garbled text, so I moved each cut to the true sentence end.
- **Marking answers right or wrong:** text matching against the official answer and listed alternatives, tolerating case, punctuation, plurals and small spelling differences. Same rules at every level.

## Results on full questions
**1. Does confidence separate right answers from wrong ones? Yes, fairly well.** Right answers averaged 0.51, wrong answers 0.26. The 200 answers scored 0.5 or higher were right **84.5%** of the time, the other 300 only **52%**. As one number (AUROC): about **75%** of the time Pinocchio gave a right answer a higher score than a wrong one (likely range 0.70 to 0.79 with 500 questions).

**2. Skipping the doubtful questions** (Figure 1): accuracy is 72.8% on the 400 most-trusted answers, 80.7% on 300 and **84.5%** on 200, against 65% for random picks; the price is answering fewer questions.

**3. Are its percentages honest? No, on full questions they run low.** It averaged 42% but 65% were right: off by about 23 points (ECE 0.23; 0 is perfectly honest). Platt scaling (learn from 100 answers with known truth, test on the other 400) cuts this to **0.064** and leaves the ranking untouched (AUROC 0.75 before and after). The lowest-score groups stay off after the fix (Figure 3).

<div class="figs">
<figure><img src="figures/abstention_curve.png"><figcaption><b>Figure 1.</b> Moving left, we answer only what Pinocchio trusts most; a larger share is right. Random skipping (orange) does nothing.</figcaption></figure>
<figure><img src="figures/prefix_effect.png"><figcaption><b>Figure 2.</b> Everything as more of the question is seen. Accuracy and Pinocchio's skill both rise; its average confidence barely moves until the full question.</figcaption></figure>
</div>

<div class="figs one"><figure><img src="figures/reliability_platt.png"><figcaption><b>Figure 3.</b> Before and after Platt scaling (full questions). Answers grouped by Pinocchio's score; height is how often each group was right; the dashed line is perfectly honest.</figcaption></figure></div>

## How much of the question is seen
| Text seen | Chatbot right | Pinocchio AUROC (likely range) | Pinocchio's average confidence | Right on the 40% Pinocchio trusts most |
|---|---|---|---|---|
| First sentence (17%) | 7.6% | 0.56 (0.46 to 0.66) | 0.27 | 9.0% |
| First two sentences (34%) | 17.2% | 0.60 (0.53 to 0.67) | 0.28 | 21.5% |
| First half (56%) | 25.2% | 0.67 (0.61 to 0.72) | 0.29 | 35.0% |
| Full question (100%) | 65.0% | 0.75 (0.70 to 0.79) | 0.42 | 84.5% |

- **Most of the chatbot's accuracy comes from the late, easier clues:** 25% after half the text, 65% after all of it. Its answer after the first sentence is word-for-word the same as its final answer on only 4% of questions (22% after half).
- **From first sentence to full question:** 292 of 500 answers went from wrong to right and only 5 went from right to wrong. Pinocchio's confidence followed: on the 292 that got fixed it rose from 0.27 to 0.50; on the 170 that were wrong both times it stayed flat (0.24 to 0.26).
- **With little text it cannot help.** At the first sentence only 38 answers are right, its scores barely separate them (the range includes 0.5), and skipping its least-trusted answers barely helps (9.0% right vs 7.6% overall). Its average confidence (27%) is far above the real hit rate (8%): only 91 answers score 0.5 or more, and just 13% of those are right. That is over-confidence, the opposite of the full-question case.

## Examples
One question per group, picked by a fixed random seed (not hand-chosen). "Trusted" means a score of 0.5 or more.

| Group (of 500) | Last clue of the question | Right answer | Chatbot said | Score | Verdict |
|---|---|---|---|---|---|
| Trusted, right (169) | "...name these great apes more closely related to humans than gorillas." | Chimpanzee | Chimpanzees | 0.57 | Works |
| Trusted, wrong (31) | "...composer of symphonies nicknamed Military, Farewell, and Surprise." | Haydn | Mozart | 0.54 | Fails: trusts a confident factual mistake |
| Doubted, right (156) | "...publications that introduced The Spirit and Superman." | Comic books | Comic books | 0.11 | Fails: a right answer would be skipped |
| Doubted, wrong (144) | "...method of academic inquiry championed by Francis Bacon..." | Science | Induction | 0.07 | Works |

**Changing as more is seen.** *Blood Wedding:* from the first sentence ("A character in this play explains her fear of 'endless bitterness' while her servant combs her hair.") the chatbot guessed Lady Macbeth (score 0.35), then Annie John (0.11), then The Sun Also Rises (0.38), and with the full question it got **Blood Wedding right, but Pinocchio gave it only 0.12**: more confident in two wrong guesses than in the final right one. *Endoplasmic reticulum* went the other way: right from the first sentence (0.76), but wrong (Lysosome) with the full question, where Pinocchio's score fell to 0.17, as it should. Only 5 questions go right-then-wrong; this is the clearest case.

- **Some "trusted but wrong" answers are a marking problem.** I read all 31 (me with AI help, two AI reviewers agreeing within a couple of rows, no independent human; labels in `results/trusted_wrong_review.csv`): 7 were clearly right in different words (e.g. "4" for "four", "UK" for "Great Britain") and 5 more were borderline. The other 19 were real confident mistakes, usually a closely related person or place (Dylan Thomas for Cecil Day-Lewis, Ontario for Quebec, Tom Sawyer for Jim). Counting the 7, accuracy on the top 200 would be about 88% instead of 84.5%. All headline numbers keep the original marking, for comparability.


## Things to know
- **Marking is not perfect.** An earlier check of 100 markings (AI-assisted, no independent person) found 7 wrong (4 too generous, 3 too strict), so true accuracy may differ a few points from 65%.
- **Telling Pinocchio which chatbot wrote the answers** (as in all runs here) changes how far off its percentages are (about 23 points with the name, 9 without), not which answers it prefers (AUROC 0.75 either way).
- **A simpler method did better:** the chatbot's own confidence picked right answers more reliably (AUROC 0.84 vs 0.75) but needs access to its inner workings.
- **Limits:** one chatbot, 500 questions, one data file; the first-sentence results rest on only 38 right answers.
