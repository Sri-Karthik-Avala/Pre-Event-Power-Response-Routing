# Pre-Event Power Response Routing — v1 findings

## The one fact that governs this challenge

`train.csv`'s 1684 rows are **183 events**, not 1684 samples. The lead-time windows of one event
overlap heavily, so they can be chained by shared 8-sample subsequences of `power_W_sequence`:

- 183 train groups (153 of them exactly 10 windows), **label-pure in 100% of groups**
- 78 test groups by the same construction (676 rows)

So `n_effective` is 183 train / 78 test. Any CV that is not grouped on this key reports fiction:
row-level random CV scores **0.994 accuracy**, the same features under event-grouped CV score
**0.305** — pure event memorisation.

## Structure of the signals

- `power_W = voltage_V * current_A * power_factor` **exactly** (relative residual 0.0000, sd 0.0000).
  Only 5 of the 6 channels are independent.
- Every channel's lag-1 autocorrelation is ~0.00. The 60 samples are i.i.d. draws around a level;
  there is no temporal structure for a sequence model to find. Windows are (level, noise scale) pairs.
- Channels are quantised: V and F and W to 0.1, I and PF to 0.001. F takes 5 distinct values total.
  `wifi_rssi` is *constant within the window* for 80% of rows, pegged at −40 / −42 / −70.
- Levels cluster into discrete device regimes: PF ∈ {0.78, 0.90, 0.95}, I spanning 1–32 A.
- Per event there is a single shared **noise multiplier** — log-noise across V, I, PF, F correlates
  0.71–0.92 across channels.

## Covariate shift, train → test

Test is two held-out calendar days, and the day-level baselines move:

| channel | train mean | test mean |
|---|---|---|
| current_A | 10.43 | 7.78 |
| power_W | 1854 | 1392 |
| wifi_rssi | −44.2 | −52.0 |

Rows with mean RSSI < −55: 10.2% of train, 37.0% of test. Absolute-level features are the ones that
shift, so nothing keys on them alone.

## What was tested, and what it scored

Event-grouped `StratifiedGroupKFold(5)`, event-weighted macro-F1 exactly as specified.

| model | score | route F1 | urgency F1 |
|---|---|---|---|
| HistGradientBoosting, 99 hand features | 0.3040 | 0.3586 | 0.2025 |
| ExtraTrees, same | 0.2883 | 0.3245 | 0.2211 |
| LogisticRegression, same | 0.2819 | 0.3109 | 0.2281 |
| 1D-CNN + aux head, 3 seeds x 5 folds | 0.2693 | 0.2943 | 0.2229 |
| ...with prior correction | 0.2739 | 0.3086 | 0.2093 |
| focused per-channel noise features, HGB | 0.3143 | 0.3477 | 0.2525 |

**The zero-information ceiling for this metric is 0.3042.** Under no signal, macro-F1 is maximised by
predicting proportionally to the class prior, which gives F1_c = p_c for every class, so macro-F1 =
1/K exactly: `0.65 * (1/3) + 0.35 * (1/4) = 0.3042`. Every model above sits on that number.

Things ruled out as label carriers, each under event-grouped CV:

- absolute levels, noise scales, ranges, percentiles, slopes, skew/kurtosis of all six channels
- cross-channel correlations and the V·I·PF consistency residual (it is identically zero)
- device identity: nearest-neighbour on (log I, PF, V, W) matches route 0.399 vs 0.343 chance,
  urgency 0.279 vs 0.273 chance; keying on rounded (PF, I, W) gives 19–27% label-pure device groups
- the per-event noise multiplier vs urgency: Kruskal p = 0.38, Spearman rho = 0.128 (p = 0.084)

The 0.3143 row is one configuration out of ten tried, is within noise of the ceiling, and should be
read as selection noise, not a finding.

## What v1 ships

`solution.py` trains, per fold, both a 1D-CNN over the raw 6-channel window (absolute-normalised and
row-shape-normalised planes, 2x6 input channels) with a summary-feature branch fused at the head, and
a HistGradientBoosting model on the summary features; blends them 50/50; 5 folds x 2 seeds bagged.
Loss is 0.65/0.35 weighted across the two heads, with event weights `1/group_size` and class-balanced
weights, so training targets the same balance the metric does.

The decode is the deliberate part. Argmax of a near-uniform posterior drifts toward whichever class
the model happens to tilt to and can zero out a class entirely, which is expensive under macro-F1.
Instead, per-class log-offsets are fitted on the out-of-fold predictions so that the OOF predicted
class rates match the train class prior, then applied to each test row independently. That is fitted
on train and applied per row — no test-set statistics — and it reaches the 0.3042 optimum under zero
signal while still passing through any real signal the model finds.

## Decisions worth your review

1. **Test-side event grouping was left on the table.** The same 8-gram chaining recovers all 78 test
   events exactly, and averaging predictions within an event would cut prediction variance. It needs
   whole-test-set visibility, which guidebook §4.2 bans outright, so I did not use it. It is also
   worth little here — averaging noise yields nothing when per-row signal is ~0.
2. **No wall-clock deadline guard**, per the Deterministic Execution finding: fixed epochs, fixed
   folds, step-based LR. Runtime is ~8 minutes on 8 threads, far inside budget, so a guard would only
   add risk.
3. **This may simply be a low-ceiling challenge.** If the returned score lands near 0.30, that is the
   information ceiling rather than a modelling failure, and the next credit should not be spent on
   more model capacity. The one hypothesis still worth testing is a per-device baseline fitted on
   train and applied per test row (deviation-from-normal rather than absolute level) — it is the only
   feature abstraction not yet ruled out.

## v1 shipped run (cold, end-to-end)

`python solution.py . working/submission.csv` — exit 0, 676 rows + header, validator: all checks passed.

- OOF route agreement 0.2969, OOF urgency agreement 0.2613
- predicted test route rate 0.444 / 0.325 / 0.231 vs train event-weighted prior 0.393 / 0.350 / 0.257
- predicted test urgency rate urgent 0.371 vs prior 0.372; all 3 routes and all 4 urgencies present,
  so no class is zeroed out under macro-F1

Compliance sweep clean: single `# made by - Karthik` comment, `sys.argv` starter pattern with no
fallback branch, no environment reads, no filesystem walk, no scoring formula in the file, device
pinned to CPU, fixed epochs and folds with no wall-clock branch.

## v3 + the permutation null (the finding that settles the challenge)

Ten independent feature abstractions were tested in parallel, 177+ configurations. **All ten agents
refuted their own hypothesis.** Best honest number anywhere: 0.3432 (per-device deviation, K=16
in-fold noise-deviation, HGB, route 0.4154).

The decisive result is not any score — it is the null controls three agents ran independently:

| null control (zero signal by construction) | score |
|---|---|
| event-level label permutation, same in-fold pipeline, 6 draws | mean 0.3130, sd 0.0192, max 0.3376 |
| pure Gaussian random features, HGB / LR | 0.3057 / 0.3057 |
| random features + best-of-3-model selection, 3 draws | 0.3165, 0.3194, **0.3630** |

So the 0.3042 theoretical ceiling is a *population* quantity. The operative bar for this
183-event grouped CV is the finite-sample null at **~0.313 +/- 0.019**, and once you take the max over
models and feature sets it reaches **0.36**. Random noise beat the 0.35 target through this harness.

Consequences:
- Every number in the table above (0.3143, 0.3193, 0.3282, 0.3320, 0.3354, 0.3459) is inside the null.
- Nothing should be believed here until it clears roughly 0.37, or is validated against its own
  permutation null.
- A 0.35 leaderboard score on 78 test events is reachable by luck alone. It would not be evidence of
  signal, and it would not be reproducible on the private LB.

v3 ships: v1's 51-feature set (the measured winner in a direct A/B — the expanded v2 feature set cost
0.023) with an ExtraTrees-dominant blend (0.50 ET / 0.30 HGB / 0.20 CNN) and the prior-matched decode.
Direct A/B on the shipped metric: v1 features 0.3114, v2 features 0.2889. OOF route agreement 0.2874,
urgency 0.2678. Validator passes; 478/676 targets match v1.
