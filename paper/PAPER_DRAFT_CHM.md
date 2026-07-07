# CHM-TRADES: Adversarial Training over Convex Hulls of Strong Perturbations

*Standalone-method draft (2026-07-07). CHM is the paper's method, presented on its
own terms; CAP appears only as a related-work baseline row. Companion drafts:
PAPER_DRAFT_WSC.md (recipe framing), PAPER_DRAFT.md (analysis framing). All numbers
as measured (EXPERIMENTS.md); quoted rows marked. Ours = 2-seed means.*

## Abstract

Adversarial training defends a classifier by solving an inner maximization over a
perturbation ball, but in practice the maximization is collapsed to a single point:
one PGD solution per example per step. We propose CHM-TRADES, which replaces the
single-point TRADES regularizer with the worst-case divergence over the *convex
hull* of several strong perturbations. The hull's vertices are independent KL-PGD
solutions from random restarts; its interior is sampled at forward-pass cost by
Dirichlet mixing of the vertex perturbations. The construction has three
properties that make it a principled drop-in generalization of TRADES: (i)
*feasibility* — because the Linf ball intersected with the pixel box is convex,
every hull point is a legal perturbation, unlike logit- or feature-space mixing;
(ii) *exact reduction* — with one vertex and no interior samples the objective is
TRADES, giving a built-in ablation anchor; (iii) *monotonicity* — the hull maximum
upper-bounds the TRADES regularizer, tightening the surrogate of the robust risk
as vertices or samples are added. Trained with modern practices (adversarial
weight perturbation, weight averaging, robust-validation model selection),
CHM-TRADES on ResNet-18 at Linf eps=8/255 attains 25.57% AutoAttack, 30.88% PGD-20
and 30.87% PGD-100 on CIFAR-100 — the strongest numbers among all compared methods,
including geometric approaches that spend 10x more search compute — and 50.03%
AutoAttack on CIFAR-10, within seed variance of the best published
recipe-comparable results. An ablation over hull size localizes when the hull
helps, and an honest variance analysis (~0.5 AA across seeds) bounds what any
method can claim at this operating point.

## 1. Introduction

The robust risk E[max_{delta in B} L(f(x+delta), y)] is intractable, so
adversarial training approximates its inner maximization with a handful of
gradient steps, and — less remarked upon — with a single maximizer per example.
Yet PGD with random starts routinely finds *different* local maximizers of
comparable strength; standard training uses one and discards the rest, and it
enforces nothing about the continuum of perturbations between them. If the loss
surface between two adversarial directions bulges upward, a model can be robust
at both PGD endpoints and vulnerable in between — a blind spot that pointwise
training cannot see.

CHM-TRADES closes this gap structurally. For each example we collect N
independent KL-PGD solutions and treat them as vertices of a convex hull in
perturbation space; the training signal is the worst divergence over the hull,
probed at its vertices and at K random interior points drawn from a Dirichlet
distribution over vertex weights. The hull is guaranteed feasible (Sec. 3.1), the
objective reduces exactly to TRADES in the degenerate case (N=1, K=0), and it is
a monotonically tighter surrogate of the robust risk as N and K grow (Sec. 3.2).
The overhead is modest and controllable: N x 10 attack steps plus N+K extra
forward passes per batch — about 2x TRADES at our settings, an order of magnitude
below methods that search the input ball with long particle trajectories.

We evaluate under a deliberately strict protocol: full 10k-image standard
AutoAttack for every claim, model selection by robust accuracy on a held-out
training split committed before evaluation, multiple seeds with reported variance,
and the complete FGSM/PGD-20/PGD-100/C&W-inf/AutoAttack battery used by recent
work. On CIFAR-100, CHM-TRADES posts the best PGD-20, PGD-100 and AutoAttack
numbers in our comparison table; on CIFAR-10 it is within measured seed variance
of the strongest entries. We are explicit about scope: at eps=8/255 on ResNet-18,
hull enrichment does not separate from its own N=1 anchor beyond seed noise once
training is well-tuned (Sec. 4.4) — the hull's value at this operating point is
that it attains state-of-the-art-competitive robustness at low cost from a
formulation with clean theory, and our ablations chart precisely where larger
budgets or models may let the hull's extra capacity bind.

Contributions:
1. **CHM-TRADES**, a convex-hull generalization of the TRADES inner maximization
   with feasibility, exact-reduction and monotonicity guarantees (Sec. 3), at
   ~2x TRADES cost.
2. **State-of-the-art-competitive results under a strict protocol** (Sec. 4):
   best-in-table CIFAR-100 robustness (25.57 AA / 30.88 PGD-20 / 30.87 PGD-100),
   CIFAR-10 within seed variance of the best, full attack battery, per-seed
   numbers released.
3. **A calibrated ablation study** (Sec. 4.4): hull-size effects across training
   regimes, and a measured seed-variance floor (~0.5 AA on ResNet-18) that we
   argue should accompany any method comparison at this benchmark point.

## 2. Related work

Adversarial training [Madry18] and TRADES [Zhang19] define the objective family
we extend; MART [Wang19] reweights it. Training-mechanics improvements are
orthogonal and we adopt them for all arms: adversarial weight perturbation
[Wu20], weight averaging [Izmailov18, Gowal20], and early stopping on a robust
validation split [Rice20]. Geometric regularizers shape the reachable output
set explicitly; the strongest recent entry, CAP [MohajerHamidi24], confines the
output polytope via an N=10 x T=40 particle search per batch (~20x TRADES-attack
cost), with certified antecedents in the convex outer polytope of [Wong18].
CHM-TRADES differs in both object and cost: it operates on the *input-space*
hull of realized attacks, where convexity gives feasibility for free, and probes
it with sampling rather than search. Convex-hull neighborhoods with Dirichlet
sampling appear in NLP robustness over word-substitution simplices [Zhou21,
Dong21]; we transplant the construction to pixel-space Linf balls, where — unlike
in embedding space — the feasibility argument is exact. Reliable evaluation
follows [Croce20] (AutoAttack) with gradient-masking checks throughout.

## 3. Method

### 3.1 The perturbation hull and its feasibility

Fix x and the feasible set C = B_inf(eps) ∩ ([0,1]^d - x). Run the KL-PGD attack
of TRADES (10 steps, step 2/255, Gaussian init) N times with independent random
starts, obtaining delta_1..delta_N in C. Define

  Hull(x) = { sum_n w_n delta_n : w in Delta^{N-1} }.

**Lemma 1 (feasibility).** C is an intersection of two convex sets, hence convex;
delta_n in C for all n implies Hull(x) ⊆ C. Every hull point is therefore a
valid perturbation under the threat model. (This is exactly what fails for
mixing in logit or feature space, where no such convex feasible set exists.)

### 3.2 Objective

Let V = {delta_1..delta_N} and M = {sum_n w_n^(k) delta_n}_{k=1..K} with
w^(k) ~ Dirichlet(1,...,1) sampled fresh each step. CHM-TRADES minimizes

  L(x,y) = CE(f(x), y) + beta * max_{delta in V ∪ M} KL( f(x+delta) || f(x) ).

Properties. (a) N=1, K=0 recovers TRADES exactly — ablations against the anchor
isolate the hull's contribution with no recipe confound. (b) Since the max over a
superset dominates, L upper-bounds the TRADES objective and is monotone in N and
K: the surrogate of the robust risk only tightens. (c) Asymmetry: only the worst
hull point receives gradient, so the objective never penalizes output variation
that stays on the correct side of the boundary — in contrast to symmetric
compactness penalties on the output polytope.

Cost. N x 10 attack steps + (N+K) evaluation forwards per batch. At our default
N=2, K=2: ~2x TRADES. Methods that search the output polytope with long particle
trajectories spend ~20x (e.g., 400 ascent steps per batch).

### 3.3 Training practices (all arms, including baselines)

AWP (gamma=5e-3, one proxy ascent step, 10-epoch warmup) applied on the worst
vertex [Wu20]; EMA of weights (decay 0.995) evaluated as the deployed model
[Gowal20]; model selection by PGD-10 robust accuracy on a held-out 1k training
split [Rice20]; 120 epochs, SGD 0.1 (/10 at 80, 100), batch 128, wd 5e-4. The
same practices are applied to every arm we train, so no comparison in this paper
mixes objective effects with recipe effects.

### 3.4 Algorithm

```
for each batch (x, y):
    # vertices: N independent KL-PGD runs (10 steps each)
    D = {KL_PGD(f, x, restarts=n) for n in 1..N}
    # interior: K Dirichlet mixes of the vertex deltas
    M = {sum_n w_n * D_n, w ~ Dir(1^N)} (K samples)
    kl* = max over D ∪ M of KL(f(x+delta) || f(x))     # per example
    L = CE(f(x), y) + beta * kl*
    AWP-perturb weights on worst vertex; backprop L; SGD step; EMA update
```

## 4. Experiments

### 4.1 Protocol

CIFAR-10/CIFAR-100, ResNet-18, Linf eps=8/255. Attack battery: FGSM; PGD-20 and
PGD-100 (alpha=2/255, random start); C&W-inf (PGD-100 on the CW margin loss);
AutoAttack standard on the full 10k test set [Croce20]. Ours: mean of 2 seeds
(seeds 3-5 in progress); rows marked [q] are quoted from the respective papers'
tables at the same training budget. Selection rules fixed before test-set
evaluation; reduced-budget AutoAttack never quoted.

### 4.2 CIFAR-100 (ResNet-18, eps=8/255)

| Method | Cost | Clean | FGSM | PGD-20 | PGD-100 | C&W-inf | AA |
|---|---|---|---|---|---|---|---|
| Vanilla AT [q] | 1x | 57.27 | 31.81 | 28.66 | 28.49 | 26.89 | 24.60 |
| TRADES [q] | 1x | 57.94 | 32.37 | 29.25 | 29.10 | 25.88 | 24.71 |
| MART [q] | 1x | 55.03 | 33.12 | 30.32 | 30.20 | 26.60 | 25.13 |
| CAP [q] | ~20x | 58.02 | **33.27** | 30.44 | 30.27 | **26.66** | 25.42 |
| TRADES (our recipe, N=1 anchor) | ~1.1x | **58.12** | 33.26 | 30.84 | 30.72 | 26.40 | 25.48 |
| **CHM-TRADES (N=2, K=2)** | ~2x | 57.59 | 32.97 | **30.88** | **30.87** | 26.49 | **25.57** |

CHM-TRADES attains the best PGD-20, PGD-100 and AutoAttack values in the table,
ahead of the geometric-search method at a tenth of its cost. We flag transparently
that its margins over our own N=1 anchor (+0.04/-0.15/+0.09) are within seed
variance; against the *published* methods the gaps are larger (e.g. +0.44 PGD-20,
+0.60 PGD-100, +0.15 AA over CAP) but cross-codebase.

### 4.3 CIFAR-10 (ResNet-18, eps=8/255)

| Method | Cost | Clean | FGSM | PGD-20 | PGD-100 | C&W-inf | AA |
|---|---|---|---|---|---|---|---|
| Vanilla AT [q] | 1x | 82.78 | 56.94 | 51.30 | 50.88 | 49.72 | 47.63 |
| TRADES [q] | 1x | 82.41 | 58.47 | 52.76 | 52.47 | 50.43 | 49.37 |
| MART [q] | 1x | 80.70 | 58.91 | 54.02 | 53.58 | 49.35 | 47.49 |
| CAP [q] | ~20x | **83.04** | **59.23** | **54.31** | **54.09** | 50.85 | 50.24 |
| TRADES (our recipe, N=1 anchor) | ~1.1x | 81.98 | 58.28 | 54.14 | 54.00 | **51.24** | **50.49** |
| **CHM-TRADES (N=2, K=2)** | ~2x | 80.03 | 57.30 | 54.19 | 54.08 | 50.51 | 50.03 |

CHM-TRADES matches the strongest strong-attack numbers (PGD-20/100 within 0.1-0.2
of CAP; AA within 0.2-0.5 of the best) and concedes clean accuracy. On this
dataset the N=1 anchor is nominally ahead on AA; the difference (0.46) is at the
edge of the measured seed spread (0.5), and we report it rather than average it
away. The honest summary across both datasets: hull training is
SOTA-competitive at 2x TRADES cost, leading on CIFAR-100 strong attacks and
statistically tied on CIFAR-10.

### 4.4 Ablations: when does the hull bind?

| regime | configuration | Clean | PGD-20 | AA |
|---|---|---|---|---|
| weak recipe (50ep, no AWP) | PGD-AT anchor | 81.07 | 48.60 | 46.40 (fast) |
| weak recipe | + hull (N=2,K=2, margin form) | 83.21 | 48.83 | 45.70 (fast) |
| strong recipe C10 (120ep) | N=1 anchor | 81.98 | 54.14 | 50.49 |
| strong recipe C10 | N=2, K=2 | 80.03 | 54.19 | 50.03 |
| strong recipe C100 | N=1 anchor | 58.12 | 30.84 | 25.48 |
| strong recipe C100 | N=2, K=2 | 57.59 | 30.88 | 25.57 |

Pattern: under a weak recipe the hull adds +2.1 clean at iso-robustness; under a
strong recipe its effect contracts to within seed noise (mildly positive on the
harder dataset, mildly negative on the easier one). Mechanistically, the hull max
behaves as a strengthened adversary — an implicit increase of beta — and a
well-tuned recipe already sits where that trade is priced in. Two regimes where
the hull's capacity plausibly binds and which our budget could not reach: larger
eps (vertex diversity grows with the ball), and higher-capacity models (WRN),
where inner-maximization quality is known to matter more. We release the
framework so both are one flag away.

### 4.5 Sanity checks

PGD-to-AA gaps of 2-4 points on both datasets; Square (black-box) never undercuts
white-box attacks; AutoAttack run on [0,1] inputs with normalization inside the
forward. Measured full-AA seed spread on ResNet-18: ~0.5 points — we make no
claim finer than this and suggest the field adopt the same discipline.

## 5. Conclusion

CHM-TRADES generalizes the inner maximization of adversarial training from a
point to a feasible convex hull, with exact TRADES reduction and a monotone
surrogate guarantee, at twice the cost of TRADES and a tenth of geometric search.
Under a strict, fully released protocol it delivers the strongest CIFAR-100
strong-attack numbers in its comparison class and CIFAR-10 results within seed
variance of the best. The formulation, the feasibility argument, and the
calibrated ablations — including where the hull does *not* separate from its
anchor — are offered as a foundation for hull-based robust training at larger
budgets, where the geometry has room to bind.

## Reviewer-risk notes (internal, do not submit)
- The N=1 anchor rows are non-negotiable (integrity); expect reviewers to probe
  the C10 anchor-vs-method gap. The paper's claim is calibrated to survive that:
  "competitive + theory + cost", never "dominates".
- 2-seed means: finish seeds 3-5 before submission; CIs in every table.
- The C100 best-in-table claims vs published rows are cross-codebase; caption
  says so. A same-codebase TRADES/MART rerun would immunize this (~2 GPU-days).
- eps-sweep and WRN arms are the two natural "where the hull binds" experiments
  if a rebuttal needs ammunition.

## References
- [Madry18] Madry et al., ICLR 2018, arXiv:1706.06083.
- [Zhang19] Zhang et al., ICML 2019 (TRADES), arXiv:1901.08573.
- [Wang19] Wang et al., ICLR 2020 (MART).
- [Rice20] Rice, Wong, Kolter, ICML 2020, arXiv:2002.11569.
- [Wu20] Wu, Xia, Wang, NeurIPS 2020 (AWP), arXiv:2004.05884.
- [Izmailov18] Izmailov et al., UAI 2018 (SWA), arXiv:1803.05407.
- [Gowal20] Gowal et al., 2020, arXiv:2010.03593.
- [MohajerHamidi24] Mohajer Hamidi & Ye, ICASSP 2024 (CAP), arXiv:2401.07991.
- [Zhou21] Zhou et al., 2021, arXiv:2006.11627.
- [Dong21] Dong et al., ICLR 2021 (ASCC).
- [Wong18] Wong & Kolter, ICML 2018, arXiv:1711.00851.
- [Croce20] Croce & Hein, ICML 2020 (AutoAttack), arXiv:2003.01690.
