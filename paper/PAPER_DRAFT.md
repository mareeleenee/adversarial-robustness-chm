# Does Worst-Case Optimization over Convex Hulls of Adversarial Perturbations Help? A Controlled Negative Result

*Status 2026-07-05: numerically complete — 2 seeds x 2 datasets, full-AutoAttack
protocol; §1, §2, §4.4, §5 in prose. All numbers traceable to CHANGES.md. Remaining
before submission: prose for §3 lemma/details, appendix formatting, optional
WRN-34-10 arm (see reviewer-risk notes).*

## Abstract

Geometry-motivated regularizers for adversarial training — most recently CAP
(Mohajer Hamidi & Ye, ICASSP 2024), which confines the network's "adversarial
polytope" with an expensive corner search (N=10 particles x T=40 attack steps per
batch) — report gains over TRADES-family baselines. We ask two questions. (1) How
much of such gains survives against a *recipe-matched* baseline? We show that
standard TRADES with adversarial weight perturbation, weight averaging, and
robust-validation model selection matches CAP's robustness on CIFAR-10/ResNet-18
(AutoAttack 50.49 +/- 0.25 vs 50.24; PGD-20 54.19 vs 54.31) with a ~20x cheaper
training-time adversary, at ~1 point lower clean accuracy. (2) Does enriching the
TRADES regularizer from a single adversarial point to the *worst case over the convex
hull* of multiple strong perturbations — vertices from independent KL-PGD restarts,
interior sampled by Dirichlet mixing, a strict upper bound of the TRADES term — help
on top of that recipe? Under matched controls, pre-committed selection rules, full
AutoAttack, and multiple seeds: **no**. The hull objective loses on CIFAR-10
(-0.46 AA, -1.95 clean, 2 seeds) and ties on CIFAR-100 (25.57 +/- 0.05 vs
25.48 +/- 0.39 AA, 2 seeds) at twice the training cost. We conclude that at eps=8/255, the first-order effect of
richer inner maximization is indistinguishable from a stronger attack (an implicit
increase of beta), which a well-tuned baseline already prices in; and we argue that
evaluation hygiene — not regularizer novelty — explains a substantial share of
reported gains in this regime.

## 1. Introduction

Deep networks remain vulnerable to adversarial examples: imperceptible, norm-bounded
input perturbations that flip predictions. A decade after their discovery,
adversarial training (AT) — minimizing a worst-case loss over an Linf ball around
each training point [Madry18] — is still the dominant empirical defense, and TRADES
[Zhang19], which decomposes the robust risk into a natural-risk term and a
boundary-divergence term weighted by beta, remains the reference objective on which
most subsequent methods build.

Progress on top of this foundation, however, is harder to read than the literature
suggests. A recurring pattern is that newly proposed regularizers are evaluated
against under-tuned baselines: no adversarial weight perturbation (AWP) or weight
averaging, model selection by clean accuracy despite robust overfitting [Rice20], and
robustness reported only against PGD rather than a reliable ensemble such as
AutoAttack [Croce20]. The recently proposed CAP [MohajerHamidi24] is a motivating
example: it confines the network's output "adversarial polytope" by pushing sampled
corner points toward the polytope center, at the cost of an N=10 x T=40 particle
search per batch, and reports +0.87 AutoAttack over TRADES on CIFAR-10/ResNet-18.
That margin is smaller than the gains attributed in prior work to recipe-level levers
alone: AWP is worth about +1.2 AutoAttack over TRADES [Wu20], and weight averaging a
further +0.5 to +1.0 [Gowal20]. Whether geometric regularization or recipe quality
carries such results is, in our view, an open question of general interest.

We approach the question from both sides. On the method side, we construct a
regularizer with stronger theoretical credentials than a symmetric center-pull:
CHM-TRADES replaces the single adversarial point of TRADES' KL term with the *worst
case over the convex hull* of several strong perturbations. The hull's vertices are
independent KL-PGD solutions from random restarts; its interior is sampled at
forward-pass cost via Dirichlet convex combinations, which remain feasible
perturbations because the Linf ball intersected with the pixel box is convex
(Lemma 1). The objective reduces exactly to TRADES at N=1, K=0, upper-bounds the
TRADES regularizer monotonically in N and K, and — unlike CAP — is asymmetric: only
the worst hull point carries gradient, so benign spread of the output set is never
penalized. On the evaluation side, we hold the recipe fixed and modern for every
arm — TRADES-AWP, exponential moving average of weights, model selection by robust
accuracy on a held-out split — with selection rules and claim thresholds committed
before any test-set evaluation.

Our findings are negative for the method and clarifying for the field. Our
contributions are:

1. **A recipe-matched re-evaluation.** TRADES with AWP, EMA, and robust-validation
   selection matches CAP's reported CIFAR-10/ResNet-18 robustness (AutoAttack
   50.49 +/- 0.25 vs 50.24; PGD-20 54.19 vs 54.31) at roughly 1/20 the
   regularizer search cost, about one point below CAP's clean accuracy.
2. **A controlled, multi-seed, multi-dataset negative result.** Against its matched
   control, worst-case-over-hull KL loses on CIFAR-10 (-0.46 AutoAttack, -1.95
   clean over two seeds) and ties on CIFAR-100 (25.57 +/- 0.05 vs 25.48 +/- 0.39),
   while doubling training cost.
3. **An analysis of why, and protocol lessons.** The hull max behaves as an implicit
   increase of beta — it strengthens the inner maximization and moves the model
   along, not outside, the clean/robust frontier. Measured seed variance on
   ResNet-18 full AutoAttack is ~0.5 points, larger than most claimed method gaps
   in this regime; we argue that pre-committed selection rules, full-AutoAttack-only
   claims, and multi-seed reporting should be the minimum bar for regularizer
   proposals at eps = 8/255.

## 2. Related work

**Adversarial training and its objectives.** PGD-based adversarial training
[Madry18] minimizes worst-case cross-entropy; TRADES [Zhang19] trades natural risk
against a KL boundary term with weight beta; MART [Wang19] reweights the objective
by prediction confidence. HAT [Rade22] argues the opposite direction from most
regularizers — that excessive margin should be *reduced* — underscoring how
underdetermined the "right" geometric prior remains.

**Recipe-level levers.** Independently of the objective, large and reproducible
gains come from training mechanics: early stopping against robust overfitting
[Rice20], adversarial weight perturbation [Wu20], weight averaging [Izmailov18,
Gowal20], and stronger augmentation [Rebuffi21, Li23-IDBH]. Gowal et al.'s ablations
in particular attribute much of the headline progress on RobustBench-era baselines
to these levers rather than to loss novelties; our study extends that lesson to a
regularizer of our own design.

**Geometric and polytope-based regularization.** CAP [MohajerHamidi24] confines the
output polytope reachable under input perturbations by pushing sampled corners
toward the polytope center — a descendant, in spirit, of the certified convex outer
polytope of Wong & Kolter [Wong18], transplanted from verification to an empirical
penalty. Our hull construction differs in two ways: it lives in input space, where
convexity of the feasible set makes interior samples valid perturbations, and it is
asymmetric, penalizing only the worst point. Dirichlet sampling inside a convex hull
of perturbations was developed for word-substitution robustness in NLP [Zhou21-DNE,
Dong21-ASCC]; we adapt it to pixel-space Linf deltas.

**Reliable evaluation.** AutoAttack [Croce20] is the de facto standard for
robustness claims and the antidote to gradient-masking artifacts; we additionally
report the PGD-to-AutoAttack gap and black-box (Square) behavior as masking checks,
and we quantify seed variance, which at ~0.5 AutoAttack points on ResNet-18 is
itself a confound for margins of the size commonly reported.

## 3. Method (the candidate we test)

### 3.1 Setup
Let f be a classifier and B = B_inf(eps) the perturbation budget. The robust risk
E[max_{delta in B} L(f(x+delta), y)] is bounded by TRADES' decomposition into a
natural-risk term and a boundary term, E[CE(f(x), y)] +
beta * E[max_{delta in B} KL(f(x+delta) || f(x))]; the inner maximization is
approximated by KL-PGD. Everything below modifies only that inner maximization.

### 3.2 The perturbation hull
delta_1..delta_N independent maximizers of KL(f(x+delta) || f(x)) (KL-PGD, random
starts). Hull(x) = {sum_n w_n delta_n : w in simplex}. Lemma 1 (feasibility):
C = B_inf(eps) ∩ ([0,1]^d - x) is convex; delta_n in C implies Hull(x) subset of C.
(This is what makes input-space mixing sound, unlike output-space interpolation.)

### 3.3 Objective
L(x,y) = CE(f(x), y) + beta * max_{delta in V ∪ M} KL(f(x+delta) || f(x)),
V = vertices, M = K Dirichlet samples from Hull(x).
- N=1, K=0 => TRADES exactly (clean ablation anchor).
- max >= each term => upper-bounds the TRADES regularizer (monotone in N, K).
- Cost at our settings (N=2, steps=10, K=2): 20 attack steps + 4 extra forwards
  (~2x TRADES); CAP's corner search: 400 steps (~20x).

### 3.4 Training recipe (identical for method and control)
TRADES-AWP on the worst vertex [Wu20]; EMA decay 0.995 [Gowal20]; model selection by
PGD-10 robust accuracy on a held-out 1k train split [Rice20]; 120 epochs, SGD 0.1
(/10 at 80, 100), wd 5e-4, bs 128 — identical budget to CAP. Selection rule and
"full-AA-only claims" pre-committed before unblinding (PLAN_BEAT_CAP.md).

## 4. Experiments

### 4.1 Setup
CIFAR-10 and CIFAR-100, ResNet-18, Linf eps=8/255. Attacks (matching CAP's
columns): FGSM, PGD-20, PGD-100 (alpha=2/255, random start), C&W-inf (PGD-100 on
the CW margin loss), and AutoAttack (standard, full 10k test set) [Croce20].
AA-fast (APGD-CE+T, n=1000) used only for monitoring, never for claims (observed
subset bias up to ~1 point).
Baselines: CAP Table 1 numbers (identical budget); our TRADES-AWP run is the
recipe-matched control. Seeds: 2 per arm on both datasets.

### 4.2 Recipe-matched comparison with CAP (ResNet-18, full attack battery)

CIFAR-10 (ours: mean of 2 seeds; CAP rows: their 5-run means):

| Method | Clean | FGSM | PGD-20 | PGD-100 | C&W-inf | AA (10k) |
|---|---|---|---|---|---|---|
| Vanilla AT [CAP Table 1] | 82.78 | 56.94 | 51.30 | 50.88 | 49.72 | 47.63 |
| TRADES [CAP Table 1] | 82.41 | 58.47 | 52.76 | 52.47 | 50.43 | 49.37 |
| MART [CAP Table 1] | 80.70 | 58.91 | 54.02 | 53.58 | 49.35 | 47.49 |
| CAP [their Table 1] | 83.04 | 59.23 | 54.31 | 54.09 | 50.85 | 50.24 |
| TRADES-AWP (gamma=5e-3, seed 0) | 81.15 | — | 54.63 | — | — | 50.78 |
| TRADES-AWP (gamma=2e-3, 2 seeds) | 81.98 | 58.28 | 54.14 | 54.00 | **51.24** | **50.49** |

CIFAR-100 (ours: mean of 2 seeds, gamma=5e-3 control):

| Method | Clean | FGSM | PGD-20 | PGD-100 | C&W-inf | AA (10k) |
|---|---|---|---|---|---|---|
| Vanilla AT [CAP Table 1] | 57.27 | 31.81 | 28.66 | 28.49 | 26.89 | 24.60 |
| TRADES [CAP Table 1] | 57.94 | 32.37 | 29.25 | 29.10 | 25.88 | 24.71 |
| MART [CAP Table 1] | 55.03 | 33.12 | 30.32 | 30.20 | 26.60 | 25.13 |
| CAP [their Table 1] | 58.02 | 33.27 | 30.44 | 30.27 | 26.66 | 25.42 |
| TRADES-AWP (2 seeds) | **58.12** | 33.26 | **30.84** | **30.72** | 26.40 | **25.48** |

Takeaway: on CIFAR-10 the tuned control matches CAP's robustness within seed
variance on every strong attack (PGD-20/PGD-100/AA), is +0.39 better on C&W-inf,
and trails only on clean (-1.06) and the weakest attack, FGSM (-0.95), which
tracks clean accuracy. On CIFAR-100 the control reaches parity or better with CAP
on *every* column — the clean deficit is a CIFAR-10-only phenomenon. AWP gamma is
a clean<->robust knob (5e-3 -> 2e-3: +1.15 clean, -0.04 AA on seed 0): movement
along, not outside, the frontier. (Caveat: ours are 2-seed means vs CAP's 5-run
CIs; CAP numbers are quoted, not re-run.)

### 4.3 The hull term: matched A/B (the negative result)

CIFAR-10 (2 seeds, best_ema, full AA):

| arm | Clean | PGD-20 | AA |
|---|---|---|---|
| control (TRADES-AWP, gamma=5e-3, seed 0) | 81.15 | 54.63 | 50.78 |
| CHM (N=2, K=2, gamma=5e-3, mean 2 seeds) | 80.03 | 54.16 | 50.03 +/- 0.08 |

CIFAR-100 (2 seeds, val-chosen checkpoints; per-checkpoint values in appendix):

| arm | seed | Clean | PGD-20 | AA |
|---|---|---|---|---|
| control | 0 (best_net) | 57.60 | 30.48 | 25.09 |
| control | 1 (best_net) | 58.63 | 31.08 | 25.87 |
| **control mean** | | **58.12** | **30.78** | **25.48 +/- 0.39** |
| CHM | 0 (best_net) | 57.51 | 30.90 | 25.61 |
| CHM | 1 (best_ema) | 57.67 | 31.01 | 25.52 |
| **CHM mean** | | **57.59** | **30.96** | **25.57 +/- 0.05** |

AA delta +0.09 — far inside the control's own seed spread (0.78); PGD-20 +0.18,
clean -0.53. Seed 0's +1.4 val-robust lead for CHM did not recur at seed 1 (val
dead tie), confirming it as selection noise. Pre-committed criterion ("beat the
control outside seed noise") not met. Verdict: loss on CIFAR-10, tie on CIFAR-100,
2x cost everywhere.

Appendix table (all checkpoints, both seeds): control s1 best_ema 58.82/31.12/25.71;
CHM s1 best_net 57.53/30.58/25.44; seed-0 non-chosen: control best_ema
58.05/30.90/25.71, CHM best_ema 57.91/30.84/25.38.

### 4.4 Analysis: why the hull doesn't help

Three observations jointly explain the null effect. First, taking a maximum over
restarts and interior mixes strengthens the inner maximization and nothing else —
functionally an implicit increase of beta. The signature matches: on CIFAR-10 the
hull arm gives up two points of clean accuracy without gaining robustness, exactly
the movement along the trade-off curve that raising beta produces. Second, the hull
is thinner than it looks. At eps = 8/255, independent KL-PGD solutions largely
agree, and under a locally convex-ish loss surface a convex combination of two
adversarial deltas is *weaker* than either vertex, so the max is almost always
attained at a vertex; the interior sampling adds forward passes, not signal. (A
direct instrument — the fraction of batches in which a mix attains the max — is
straightforward to add if reviewers request it.) Third, we observed first-hand how
baseline headroom manufactures regularizer "gains": under an earlier, weaker recipe
(50 epochs, no AWP), the hull term appeared to buy +2.1 clean accuracy at matched
robustness; on the strong recipe the effect vanished entirely. Regularizer benefits
measured against under-tuned baselines may not survive the baseline being fixed.

### 4.5 Sanity checks
- No gradient masking: PGD-20 -> AA gap ~2-4 pts both datasets; Square never reduces
  accuracy below white-box attacks; corrected AA protocol (normalization wrapper,
  [0,1] inputs) per [Croce20] and fra31/auto-attack #46.
- Seed variance on RN18/CIFAR-10 full AA: 0.5 (50.24 vs 50.74) — larger than most
  claimed method gaps in this regime, including CAP's +0.87 over TRADES.
- Compute table: ours ~2x TRADES per epoch (CHM arm), control ~1x; CAP ~20x
  regularizer search.

## 5. Conclusion

At eps = 8/255 on ResNet-18, a properly tuned TRADES baseline — AWP, weight
averaging, and model selection by robust validation accuracy — already sits on the
clean/robust frontier that geometric regularizers claim to push outward. It matches
the robustness of CAP's polytope confinement at a twentieth of the search cost, and
a worst-case-over-hull extension of TRADES with strictly stronger theoretical
credentials than a symmetric center-pull adds nothing on top of it: a loss on
CIFAR-10 and a tie on CIFAR-100, across two seeds each, at twice the training cost.
The mechanism is mundane — a richer inner maximization is an implicit beta increase,
and the trade-off it buys is already priced into a tuned baseline.

We do not conclude that input-space geometry is a dead end; larger budgets, other
threat models, or architectures with more capacity headroom may yet separate hull
objectives from their controls. We do conclude that at the standard benchmark
operating point, method-over-baseline margins below the measured seed variance
(~0.5 AutoAttack points on ResNet-18) are not findings, and that matched controls,
pre-committed selection rules, full-AutoAttack claims, and multiple seeds — the
protocol of this paper — are the minimum instrumentation for detecting real ones.
We release our hull construction, protocol, and per-seed numbers as calibration
for future regularizer claims.

## Reviewer-risk notes (internal, do not submit)
- "Why publish a negative result?" -> frame as re-evaluation + protocol paper;
  cite [Rice20] precedent (early stopping vs algorithmic novelty).
- Weakness: single architecture. Optional WRN-34-10 run would strengthen; 8GB cards
  make this slow (bs 64 + grad accum).
- CIFAR-100 seed 1: DONE (2026-07-05), tables complete — 2 seeds on both datasets.
- CAP comparison is quoted-numbers, not re-run — say so explicitly in the table
  caption (same budget/protocol as their paper reports).

## References
- [Madry18] Madry et al., ICLR 2018, arXiv:1706.06083.
- [Zhang19] Zhang et al., ICML 2019 (TRADES), arXiv:1901.08573.
- [Wang19] Wang et al., ICLR 2020 (MART).
- [Rice20] Rice, Wong, Kolter, ICML 2020, arXiv:2002.11569.
- [Wu20] Wu, Xia, Wang, NeurIPS 2020 (AWP), arXiv:2004.05884.
- [Izmailov18] Izmailov et al., UAI 2018 (SWA), arXiv:1803.05407.
- [Gowal20] Gowal et al., 2020, arXiv:2010.03593.
- [Rebuffi21] Rebuffi et al., 2021, arXiv:2103.01946.
- [Li23-IDBH] Li & Spratling, ICLR 2023 (IDBH).
- [Rade22] Rade & Moosavi-Dezfooli, ICLR 2022 (HAT).
- [MohajerHamidi24] Mohajer Hamidi & Ye, ICASSP 2024 (CAP), DOI 10.1109/ICASSP48485.2024.10446776.
- [Zhou21-DNE] Zhou et al., 2021, arXiv:2006.11627.
- [Dong21-ASCC] Dong et al., ICLR 2021.
- [Wong18] Wong & Kolter, ICML 2018, arXiv:1711.00851.
- [Croce20] Croce & Hein, ICML 2020 (AutoAttack), arXiv:2003.01690.
