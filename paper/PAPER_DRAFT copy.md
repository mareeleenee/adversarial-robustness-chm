# Worst-Case Optimization over Convex Hulls of Adversarial Perturbations (CHM-TRADES)

*Draft skeleton — numbers marked TBD are filled from the v3 runs (see PLAN_BEAT_CAP.md
for the protocol; do not fill from single-seed or AA-fast-1000 results).*

## Abstract (draft)

Adversarial training methods that regularize the geometry of the network's reachable
output set — most recently CAP (Mohajer Hamidi & Ye, ICASSP 2024), which confines the
"adversarial polytope" by pushing sampled corner points toward its center — improve
robustness but pay a heavy search cost (hundreds of attack steps per batch) for a
symmetric compactness prior that is only loosely tied to the robust risk. We propose
CHM-TRADES: instead of confining the output polytope, we minimize the *worst-case*
KL divergence between the clean prediction and predictions over the *convex hull* of
multiple strong perturbations in input space — the hull's vertices are independent
KL-PGD solutions and its interior is sampled with Dirichlet convex combinations at
negligible cost. The resulting objective reduces exactly to TRADES with one vertex and
no interior samples, upper-bounds it otherwise, and is a direct surrogate of the robust
risk. Combined with a modern training recipe (adversarial weight perturbation, weight
averaging, robust-validation model selection), CHM-TRADES reaches **TBD%** AutoAttack
robust accuracy at **TBD%** clean accuracy with ResNet-18 on CIFAR-10 (Linf, 8/255),
surpassing CAP (50.24% / 83.04%) at a 20x lower regularizer search cost.

## 1. Introduction

- Adversarial examples; AT as the dominant empirical defense [Madry18].
- Two regularizer families: attack-loss minimization [Madry18, Wang19-MART] vs
  clean+regularizer [Zhang19-TRADES, Rade22-HAT, CAP24].
- CAP's idea and its two weaknesses (our hook):
  (a) symmetric center-pull penalizes *all* spread of the polytope, including spread
      that never crosses a decision boundary — over-regularization that the robust
      risk does not require;
  (b) corner search costs N=10 particles x T=40 steps per batch.
- Our claim: the right object is not a compact output polytope but a small *worst-case*
  divergence over a richer input neighborhood. Multiple restarts give hull vertices for
  free diversity; Dirichlet mixes cover the interior at forward-pass cost; the max over
  them tightens the TRADES bound.
- Contributions: (1) CHM-TRADES objective + feasibility argument (Linf ball ∩ [0,1] box
  is convex); (2) exact-TRADES reduction → clean ablations; (3) SOTA-recipe evaluation
  beating CAP on ResNet-18/CIFAR-10 with honest, multi-seed, full-AutoAttack protocol.

## 2. Related work

AT [Madry18]; TRADES [Zhang19]; MART [Wang19]; robust overfitting & early stopping
[Rice20]; AWP [Wu20]; weight averaging [Izmailov18, Gowal20]; augmentation for AT
[Rebuffi21, Li23-IDBH]; HAT [Rade22]; CAP [MohajerHamidi24]; hull/Dirichlet
neighborhoods in NLP [Zhou21-DNE, Dong21-ASCC]; certified polytope bounds
[Wong18-convex-outer-polytope] (the *certified* ancestor of CAP's object — cite to
position both CAP and us).

## 3. Method

### 3.1 Setup
Robust risk; TRADES decomposition: natural risk + boundary term.

### 3.2 The perturbation hull
δ_1..δ_N independent maximizers of KL(f(x+δ) || f(x)) (KL-PGD, random starts).
Hull(x) = {Σ w_n δ_n : w ∈ Δ^{N-1}}. Lemma 1 (feasibility): C = B_∞(ε) ∩ ([0,1]^d − x)
is convex, δ_n ∈ C ⇒ Hull(x) ⊆ C. (One line proof; this is what makes input-space
mixing sound, unlike output-space interpolation.)

### 3.3 Objective
L(x,y) = CE(f(x), y) + β · max_{δ ∈ V ∪ M} KL(f(x+δ) || f(x)),
V = vertices, M = K Dirichlet samples from Hull(x).
- N=1, K=0 ⇒ TRADES exactly.
- max ≥ each term ⇒ upper-bounds the TRADES regularizer (monotone in N, K).
- Asymmetry vs CAP: only the worst hull point carries gradient — no penalty on
  benign spread. Cost: N x steps attack + (N+K) extra forwards, vs CAP's N x T
  corner search (10x40). At our settings (N=2, steps=10, K=2): 20 attack steps,
  ~2x TRADES; CAP: 400.

### 3.4 Training recipe
TRADES-AWP weight perturbation on the worst vertex [Wu20]; EMA (decay 0.995)
[Gowal20]; selection by PGD-10 robust accuracy on a held-out 1k train split [Rice20];
120 epochs, SGD 0.1 (/10 at 80, 100), wd 5e-4, bs 128 — identical budget to CAP.

## 4. Experiments

### 4.1 Setup
CIFAR-10 (and CIFAR-100 if time; SVHN optional), ResNet-18, Linf ε=8/255.
Attacks: FGSM, PGD-20/100 (α=2/255), C&W∞, AutoAttack (standard, full test set).
Baselines: numbers quoted from CAP paper Table 1 (identical training budget);
our own TRADES-AWP run is the recipe-matched control.

### 4.2 Main table (ResNet-18, CIFAR-10)

| Method | Clean | PGD-20 | AA |
|---|---|---|---|
| Vanilla AT [CAP Table 1] | 82.78 | 51.30 | 47.63 |
| TRADES [CAP Table 1] | 82.41 | 52.76 | 49.37 |
| MART [CAP Table 1] | 80.70 | 54.02 | 47.49 |
| CAP [their Table 1] | 83.04 | 54.31 | 50.24 |
| TRADES-AWP (our recipe control) | TBD | TBD | TBD |
| **CHM-TRADES (ours)** | TBD | TBD | TBD |

### 4.3 Ablations
(a) N=1,K=0 (=TRADES) vs N=2,K=0 (restarts only) vs N=2,K=2 (hull interior) — isolates
the hull's contribution; (b) no AWP; (c) EMA vs best_net; (d) β sweep; (e) seeds ≥2,
report mean±std.

### 4.4 Sanity checks (reviewers will ask)
- No gradient masking: PGD-20→AA gap ~2 pts; black-box (Square) ≥ white-box; loss
  surface smoothness optional.
- Compute table: wall-clock/epoch vs TRADES, CAP (estimated from their N,T).

## 5. Conclusion (draft)
Worst-case-over-hull regularization is a cheaper, better-aligned alternative to
output-polytope confinement; recipe quality (AWP/WA/selection) accounts for the rest.

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
