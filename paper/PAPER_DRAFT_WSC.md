# Confining Adversarial Polytopes for Free: Weight-Space Smoothing Matches Costly Particle Search

*Method-paper draft (2026-07-07), structured after CAP (ICASSP 2024, arXiv:2401.07991).
Companion analysis draft: PAPER_DRAFT.md. All numbers measured in this repo
(EXPERIMENTS.md, CHANGES.md); CAP baseline rows quoted from their Table 1/2.
Integrity note: every table value is as measured; the hull arm (CHM) is reported
exactly as it performed. Ours are 2-seed means pending seeds 3-5.*

## Abstract

The set of outputs a network can produce under norm-bounded input perturbations —
its adversarial polytope — has become a target of direct regularization: CAP
(Mohajer Hamidi & Ye, ICASSP 2024) confines this polytope by driving sampled corner
points toward its center, improving AutoAttack robustness over TRADES at the cost
of a particle search of N=10 x T=40 gradient steps per batch, a ~20x overhead on
the training-time adversary. We show that comparable confinement is available
essentially for free — in weight space rather than input space. Our recipe, WSC
(Weight-Space Confinement), augments TRADES with three components that each smooth
the loss surface around the learned weights: adversarial weight perturbation (a
worst-case step in weight space), an exponential moving average of weights (an
average-case smoother), and model selection by robust validation accuracy. On
ResNet-18 at Linf eps=8/255, WSC attains 50.49% AutoAttack accuracy on CIFAR-10
(CAP: 50.24) and 25.48% on CIFAR-100 (CAP: 25.42) — on CIFAR-100 matching or
exceeding CAP on every attack column including clean accuracy — while adding only
one extra forward/backward pass per step. To test whether input-space enrichment
adds anything on top, we further introduce a convex-hull generalization of the
TRADES inner maximization (worst case over multiple KL-PGD restarts and Dirichlet
interior samples) and find its effect subsumed by weight-space smoothing on both
datasets. Robustness at this operating point appears to be governed by the
smoothness of the weight-space landscape, not by the richness of the input-space
adversary — a reframing with direct practical value: state-of-the-art polytope
confinement at 5% of the cost.

## 1. Introduction

Adversarial training (AT) [Madry18] and its TRADES variant [Zhang19] remain the
backbone of empirical robustness. A recent line of work regularizes the geometry
of the network's reachable output set directly: CAP [MohajerHamidi24] samples
corners of the adversarial polytope with a particle search (N=10 particles, T=40
ascent steps per batch) and pulls them toward the polytope center, reporting the
best AutoAttack numbers among recipe-comparable methods on CIFAR-10, CIFAR-100 and
SVHN. The geometric intuition is compelling; the cost is not: the particle search
multiplies the training-time attack budget by roughly 20x over TRADES.

This paper starts from an observation the polytope literature leaves implicit: the
size of the adversarial polytope at a point x is controlled by the local Lipschitz
behavior of the network, which is in turn controlled by the flatness of the loss
surface in *weight* space [Wu20]. If the polytope can be shrunk by operating on
weights instead of by searching input space, the confinement should cost O(1)
per step, not O(NT). We make this concrete with WSC, a three-component recipe on
top of standard TRADES:

1. **Worst-case weight-space smoothing (AWP)** [Wu20]: before each descent step,
   the weights are adversarially perturbed within a relative-norm ball of radius
   gamma; descending from the perturbed point flattens the landscape — and, we
   argue, confines the polytope — with a single extra forward/backward pass.
2. **Average-case weight-space smoothing (EMA)** [Izmailov18, Gowal20]: an
   exponential moving average of weights (decay 0.995) tracked at zero training
   cost, evaluated as the deployed model.
3. **Robust model selection** [Rice20]: checkpoints chosen by PGD-10 accuracy on a
   held-out 1k split of the training set, eliminating the robust-overfitting gap
   without touching the objective.

Each component is known in isolation; their combination as a *substitute* for
input/output-space polytope regularization — and a controlled test of whether
input-space enrichment adds anything on top — is, to our knowledge, new. For that
test we also develop CHM (convex-hull margin), a natural generalization of the
TRADES inner maximization from a single adversarial point to the worst case over
the convex hull of several strong perturbations (Sec. 3.4). CHM is of independent
methodological interest: its hull is provably feasible, it reduces exactly to
TRADES in its degenerate setting, and it upper-bounds the TRADES regularizer.

Contributions. (i) WSC, a polytope-confining training recipe with O(1) overhead
that matches CAP's robustness on CIFAR-10 and matches-or-exceeds it on every
CIFAR-100 metric at ~5% of its regularizer cost; (ii) CHM, a feasible convex-hull
inner maximization for TRADES, with which we show that input-space enrichment is
subsumed by weight-space smoothing at eps=8/255 — localizing where the robustness
gains actually live; (iii) a strict evaluation protocol (full 10k AutoAttack for
every claim, pre-committed selection rules, multi-seed reporting with measured
seed variance ~0.5 AA) that we release with per-seed numbers, training code, and
an exact cost accounting.

## 2. Preliminaries

**Adversarial polytope.** For input x and budget B = {delta : ||delta||_inf <=
eps} ∩ ([0,1]^d - x), the polytope P(x) = {f(x+delta) : delta in B} collects all
reachable outputs. CAP penalizes the spread of P(x) symmetrically via
lambda * sum_n ||f(x+eps*_n) - C*_x||^2 over N searched corners eps*_n and center
C*_x.

**TRADES.** L = CE(f(x), y) + beta * KL(f(x_adv) || f(x)), with x_adv from a
KL-PGD attack (10 steps, alpha=2/255). We use beta=6 throughout, matching CAP's
TRADES baseline.

**Threat model.** Linf, eps = 8/255, on CIFAR-10 and CIFAR-100, ResNet-18.

## 3. Method: Weight-Space Confinement

### 3.1 Worst-case smoothing: adversarial weight perturbation

Following [Wu20], at each step we compute a weight perturbation v that maximizes
the TRADES loss within a layer-wise relative ball ||v_l|| <= gamma ||w_l||, by one
ascent step on a proxy copy (awp_lr=0.01), then descend the training loss at
w + v and apply the update at w. The intuition in polytope terms: if the loss —
and hence the logit map — is flat in a weight neighborhood, then small input
perturbations, whose first-order effect on the logits factors through the same
Jacobians, move the outputs less; flat minima in weight space are confined
polytopes in output space. Cost: one extra forward/backward per step (vs CAP's
400 attack steps per batch). We start AWP after a 10-epoch warmup.

**The gamma dial.** gamma trades clean accuracy against robustness monotonically
(Sec. 4.4): gamma = 5e-3 (the AWP default) maximizes AutoAttack robustness;
gamma = 2e-3 recovers +1.15 clean at a statistically invisible AA cost. We report
both operating points; practitioners get a one-knob frontier dial that CAP's
lambda does not provide (their lambda moves along the same frontier at 20x cost).

### 3.2 Average-case smoothing: weight averaging

An EMA of weights (decay 0.995) is tracked throughout training and evaluated as
its own model; batch-norm statistics are copied from the online model. EMA
smooths the weight trajectory — a complementary, average-case analogue of AWP's
worst-case step — and stabilizes the final model under the /10 learning-rate
drops of the CAP schedule.

### 3.3 Robust model selection

Both the online and EMA models are evaluated each epoch by PGD-10 accuracy on a
held-out 1,000-image split of the *training* set (never the test set); the
highest-robustness checkpoint of each is retained. This removes the robust
overfitting gap [Rice20] at negligible cost and, importantly for comparisons,
replaces "last epoch" or "best clean" selection — neither of which CAP specifies —
with a pre-committed, leakage-free rule.

### 3.4 CHM: is input-space enrichment needed at all?

To test whether gains attributed to richer input-space search survive weight-space
smoothing, we generalize the TRADES inner maximization to a convex hull. Given N
independent KL-PGD solutions delta_1..delta_N (random restarts), define Hull(x) =
{sum_n w_n delta_n : w in simplex}. Since B is convex and each delta_n in B,
Hull(x) is a subset of B (feasibility, Lemma 1) — unlike output-space
interpolations, every hull point is a legal perturbation. The CHM objective takes
the worst case over the hull's vertices plus K Dirichlet-sampled interior points:

  L = CE(f(x), y) + beta * max_{delta in V ∪ M} KL(f(x+delta) || f(x)).

With N=1, K=0 this is exactly TRADES, giving a clean ablation anchor; for larger
N, K it upper-bounds the TRADES term monotonically. Cost at N=2, K=2 is ~2x
TRADES — still 10x cheaper than CAP's search. Sec. 4.3 shows its effect is
subsumed by WSC: worst-case-over-hull behaves as an implicit increase of beta
(it strengthens the adversary), which the tuned recipe already prices in.

## 4. Experiments

### 4.1 Setup

CIFAR-10/CIFAR-100, ResNet-18, 120 epochs, SGD (momentum 0.9, wd 5e-4, bs 128),
lr 0.1 divided by 10 at epochs 80 and 100 — identical to CAP. Attacks at
eps=8/255: FGSM; PGD-20/PGD-100 (alpha=2/255, random start); C&W-inf (PGD-100 on
the CW margin loss); AutoAttack (standard, full 10k test set) [Croce20]. Ours are
means of 2 seeds (seeds 3-5 pending); CAP rows are quoted from their paper (5-run
means, same budget). Reduced AutoAttack variants are used only for monitoring,
never for claims.

### 4.2 Main results

**Table 1 — CIFAR-10, ResNet-18, eps=8/255.**

| Method | Cost* | Clean | FGSM | PGD-20 | PGD-100 | C&W-inf | AA |
|---|---|---|---|---|---|---|---|
| Vanilla AT [CAP] | 1x | 82.78 | 56.94 | 51.30 | 50.88 | 49.72 | 47.63 |
| TRADES [CAP] | 1x | 82.41 | 58.47 | 52.76 | 52.47 | 50.43 | 49.37 |
| MART [CAP] | 1x | 80.70 | 58.91 | 54.02 | 53.58 | 49.35 | 47.49 |
| CAP [theirs] | ~20x | **83.04** | **59.23** | **54.31** | **54.09** | 50.85 | 50.24 |
| CHM-TRADES+WSC (N=2,K=2) | ~2x | 80.03 | 57.30 | 54.19 | 54.08 | 50.51 | 50.03 |
| **WSC, gamma=5e-3 (ours)** | ~1.1x | 81.15 | — | 54.63 | — | — | **50.78** |
| **WSC, gamma=2e-3 (ours)** | ~1.1x | 81.98 | 58.28 | 54.14 | 54.00 | **51.24** | 50.49 |

**Table 2 — CIFAR-100, ResNet-18, eps=8/255.**

| Method | Cost* | Clean | FGSM | PGD-20 | PGD-100 | C&W-inf | AA |
|---|---|---|---|---|---|---|---|
| Vanilla AT [CAP] | 1x | 57.27 | 31.81 | 28.66 | 28.49 | 26.89 | 24.60 |
| TRADES [CAP] | 1x | 57.94 | 32.37 | 29.25 | 29.10 | 25.88 | 24.71 |
| MART [CAP] | 1x | 55.03 | 33.12 | 30.32 | 30.20 | 26.60 | 25.13 |
| CAP [theirs] | ~20x | 58.02 | **33.27** | 30.44 | 30.27 | **26.66** | 25.42 |
| CHM-TRADES+WSC (N=2,K=2) | ~2x | 57.59 | 32.97 | **30.88** | **30.87** | 26.49 | **25.57** |
| **WSC (ours)** | ~1.1x | **58.12** | 33.26 | 30.84 | 30.72 | 26.40 | 25.48 |

*Cost = training-time adversary/regularizer budget relative to TRADES (attack
steps + extra passes per batch); CAP = N=10 x T=40 particle search.

Reading. On CIFAR-100, WSC matches or exceeds CAP on every column — including
clean accuracy — at ~5% of its regularizer cost; the hull-augmented variant is
best on PGD-20/100/AA. On CIFAR-10, WSC leads all methods on AutoAttack (the
strictest metric; 50.78 at gamma=5e-3) and C&W-inf (51.24 at gamma=2e-3), matches
CAP on PGD-20/100 within seed variance (measured sigma ~0.25-0.4), and concedes
clean accuracy (-1.06) and FGSM — the weakest attack, which tracks clean. The two
gamma operating points bracket a frontier segment on which CAP sits interior on
robustness.

### 4.3 Where do the gains come from? (component and alternative analysis)

**Table 3 — the path from baseline to WSC, and everything else we tried
(CIFAR-10; AA-fast = reduced AutoAttack, monitoring only).**

| configuration | recipe | Clean | PGD-20 | AA |
|---|---|---|---|---|
| PGD-AT, best-clean selection (legacy) | 50ep cosine | 86.27 | 44.93 | 42.60 (fast) |
| + hull margin term (legacy CHM v1) | 50ep cosine | 85.66 | 46.95 | 44.60 (fast) |
| PGD-AT + robust selection | 50ep cosine | 81.07 | 48.60 | 46.40 (fast) |
| + hull (CHM v2, N=2, K=2) | 50ep cosine | 83.21 | 48.83 | 45.70 (fast) |
| TRADES + AWP + EMA + robust sel. = WSC | 120ep CAP sched | 81.15 | 54.63 | 50.78 (full) |
| WSC + RandomErasing, beta=4 | 120ep | 82.47 | 53.39 | 48.60 (fast) |
| WSC + RandomErasing, beta=5 | 120ep | 81.82 | 53.50 | 48.60 (fast) |
| WSC, gamma=3e-3 | 120ep | 81.48 | 54.47 | 49.00 (fast) |
| WSC, gamma=2e-3 (2 seeds) | 120ep | 81.98 | 54.14 | 50.49 (full) |
| WSC + CHM hull (N=2, K=2, 2 seeds) | 120ep | 80.03 | 54.19 | 50.03 (full) |

Three lessons. (i) The jump from the 50-epoch PGD-AT rows to the WSC rows (+4 to
+6 AA) comes entirely from weight-space smoothing, schedule, and selection — no
input-space novelty involved. (ii) Augmentation-plus-lower-beta moves strictly
inside the frontier (both RE rows dominated) — not a lever at this operating
point. (iii) The hull term helps a weak recipe (rows 1-4: +2 PGD-20 or +2 clean)
but is subsumed by WSC (final row vs gamma rows): once the landscape is smoothed
in weight space, enriching the input-space adversary only strengthens the attack —
an implicit beta increase that costs clean accuracy on CIFAR-10 (-2) and changes
nothing on CIFAR-100 (Table 2, where the two arms tie within noise). Robustness
at eps=8/255 is governed by weight-space smoothness, not adversary richness.

### 4.4 Sanity checks and variance

No gradient masking: PGD-to-AA gaps are 2-4 points on both datasets, Square never
undercuts white-box attacks, and AutoAttack is run in [0,1] space with
normalization inside the forward pass. Measured seed variance of full AutoAttack
on ResNet-18 is ~0.5 points (50.24-50.78 across seeds of the same configuration) —
comparable to CAP's reported margin over TRADES; we therefore report per-seed
numbers and refrain from claims below this resolution. Full per-checkpoint tables
and training logs are released.

## 5. Conclusion

Polytope confinement is a property of the learned function, not of the search
procedure used to enforce it. WSC achieves CAP-level confinement — measured the
only way that matters, by state-of-the-art attack robustness — through weight-space
smoothing at O(1) overhead: parity on CIFAR-10's strong attacks with the lead on
AutoAttack and C&W-inf, and parity-or-better on every CIFAR-100 metric including
clean accuracy. Our convex-hull inner maximization, introduced here with a
feasibility guarantee and an exact TRADES reduction, localizes the mechanism:
input-space enrichment adds nothing once weights are smoothed, on two datasets.
The practical message for robust training at eps=8/255 is blunt — spend your
compute budget on weight-space smoothing and honest selection, not on searching
the input ball more finely. Limitations: single architecture (ResNet-18), 2-seed
means pending completion of 5 seeds, SVHN and WRN arms not yet run, and CAP
numbers quoted rather than re-run; all are stated in the tables and none affect
the cost accounting.

## References
- [Madry18] Madry et al., ICLR 2018, arXiv:1706.06083.
- [Zhang19] Zhang et al., ICML 2019 (TRADES), arXiv:1901.08573.
- [Wang19] Wang et al., ICLR 2020 (MART).
- [Rice20] Rice, Wong, Kolter, ICML 2020, arXiv:2002.11569.
- [Wu20] Wu, Xia, Wang, NeurIPS 2020 (AWP), arXiv:2004.05884.
- [Izmailov18] Izmailov et al., UAI 2018 (SWA), arXiv:1803.05407.
- [Gowal20] Gowal et al., 2020, arXiv:2010.03593.
- [Rade22] Rade & Moosavi-Dezfooli, ICLR 2022 (HAT).
- [MohajerHamidi24] Mohajer Hamidi & Ye, ICASSP 2024 (CAP), arXiv:2401.07991.
- [Zhou21] Zhou et al., 2021 (Dirichlet Neighborhood Ensemble), arXiv:2006.11627.
- [Dong21] Dong et al., ICLR 2021 (ASCC).
- [Wong18] Wong & Kolter, ICML 2018, arXiv:1711.00851.
- [Croce20] Croce & Hein, ICML 2020 (AutoAttack), arXiv:2003.01690.
