# bf16 vs FP8

120 questions in both runs. Graded by the 'correct' field in each run.

1 flip out of 120 is consistent with no real difference (exact McNemar p = 1.0), though not evidence that there is none; you need N ≈ 790 questions to detect a 1-point drop (80% power, exact McNemar, α = 0.05) if every flip goes the same way (the fewest possible), and about 3,745 if 4.6% of answers flip (the upper end of the 95% interval for the flip rate seen here).
Accuracy difference bf16 − FP8: +0.8 points, 95% interval [-2.3, +4.6]. The data are compatible with FP8 being anywhere from 2.3 points better to 4.6 points worse than bf16; drops larger than 4.6 points fall outside the interval, smaller ones do not.
With fewer than 6 flips no split can reach p ≤ 0.05, even if every flip goes the same way.

## Accuracy

| run | correct |
|---|---|
| bf16 | 72/120 = 60.0% [51.1%, 68.3%] |
| FP8 | 71/120 = 59.2% [50.2%, 67.5%] |

## Paired table

| | FP8 right | FP8 wrong |
|---|---|---|
| **bf16 right** | 71 (both right) | **1** (only bf16: b) |
| **bf16 wrong** | **0** (only FP8: c) | 48 (both wrong) |

- discordant pairs: b = 1, c = 0 (flip rate 0.8%)
- exact McNemar (binomial, two-sided): p = 1.0
- accuracy difference bf16 − FP8: +0.8 points, 95% CI [-2.3, +4.6] (Tango score interval, paired)

## Same answers

| measure | value |
|---|---|
| word-for-word identical (exact text, ignoring leading/trailing spaces) | 49/120 = 40.8% [32.5%, 49.8%] |
| identical after normalising case, punctuation, articles | 51/120 = 42.5% [34.0%, 51.4%] |
| same answer by token overlap (F1 between the two >= 0.5, or both abstain) | 109/120 = 90.8% [84.3%, 94.8%] |
| **both wrong when the answers are the same (token overlap)** | 42/109 = 38.5% [29.9%, 47.9%] |
| both wrong when word-for-word identical | 18/49 = 36.7% [24.7%, 50.7%] |
| same wrong answer (token overlap), among questions both got wrong | 42/48 = 87.5% [75.3%, 94.1%] |

Token overlap is a cheap proxy for 'the same answer'; two answers sharing half their words can still differ in meaning. Word-for-word identity is strict.

## How many questions to see a small drop

Planning estimates under assumed flip rates, not guarantees: independent questions like these, a true drop of the given size, exact McNemar, α = 0.05, power 80%, FP8 worse than bf16 by the given points. The flip rate is the share of questions where the two runs disagree on right/wrong: observed 0.8%, upper end of its 95% Wilson interval 4.6%. A drop of x points needs at least x% of questions to flip.

| drop | every flip one way (best case) | at observed flip rate | at upper end of flip-rate interval |
|---|---|---|---|
| 1 pt | 790 | rate below the drop | 3,745 |
| 2 pt | 394 | rate below the drop | 950 |
| 3 pt | 263 | rate below the drop | 417 |

## By category

| category | n | bf16 | FP8 | both right | only bf16 | only FP8 | both wrong | McNemar p | bf16 − FP8, points [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| direct | 15 | 66.7% | 66.7% | 10 | 0 | 0 | 5 | 1.0 | +0.0 [-20.4, +20.4] |
| keyword | 15 | 66.7% | 66.7% | 10 | 0 | 0 | 5 | 1.0 | +0.0 [-20.4, +20.4] |
| multi_hop | 15 | 33.3% | 33.3% | 5 | 0 | 0 | 10 | 1.0 | +0.0 [-20.4, +20.4] |
| near_identical | 15 | 60.0% | 60.0% | 9 | 0 | 0 | 6 | 1.0 | +0.0 [-20.4, +20.4] |
| paraphrased | 15 | 40.0% | 40.0% | 6 | 0 | 0 | 9 | 1.0 | +0.0 [-20.4, +20.4] |
| plain_language | 15 | 40.0% | 33.3% | 5 | 1 | 0 | 9 | 1.0 | +6.7 [-15.1, +29.8] |
| unanswerable | 15 | 100.0% | 100.0% | 15 | 0 | 0 | 0 | 1.0 | +0.0 [-20.4, +20.4] |
| verbose | 15 | 73.3% | 73.3% | 11 | 0 | 0 | 4 | 1.0 | +0.0 [-20.4, +20.4] |

Per-category tests are not corrected for multiple comparisons.

## Flipped questions

- only bf16 right: `attention-plain_language-19`

