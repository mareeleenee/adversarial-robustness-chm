# Ideas to beat all current methods (compiled 2026-07-07, session 6)

Baseline to beat (ours, measured): **RN18, CIFAR-10, eps=8/255, full AA 50.49
(2-seed mean; best seed 50.78)**; CIFAR-100 AA 25.48/25.57. CAP: 50.24 / 25.42.
Everything below is ranked by (expected AA gain on RN18-class models) x (feasibility
on 2x RTX 2080 8GB). "Reported" = published number for RN18/PreActRN18-class models;
"expected" = my estimate for our setup — treat estimates as hypotheses, not facts.

## Tier 1 — known to give large gains; not yet in our repo

### 1. Diffusion-generated training data (the single biggest lever, +5 to +8 AA)
Train on CIFAR + 1M+ synthetic images from a diffusion model. This dominates every
objective-level trick in the literature by a wide margin.
- Wang et al., ICML 2023 (arXiv:2302.04638, code+data: github.com/wzekai99/DM-Improves-AT)
  release pre-generated 1M-50M EDM datasets; WRN-28-10 reaches 67.31 AA (C10),
  38.83 (C100). Gowal et al. 2021 (arXiv:2110.09468) report PreActRN18 ~58-59 AA
  with 100M DDPM samples; ~55-56 with 1M.
- Feasibility here: download their released 1M npz; extend main_adv_CHM_v3.py with
  a mixed loader (unsup fraction 0.7, per their recipe); 200 epochs bs128 on our
  cards ~2-3 days/run. Expected on RN18: **AA 54-57** — beats everything in all our
  tables including CAP by 4+.
- Purity caveat for the paper: this changes the "no extra data" category. Report as
  a separate track.

### 2. Robust distillation from a public robust teacher (+1.5 to +2.5 AA, CHEAP)
Distill RN18 student from an adversarially-trained WRN teacher checkpoint
(RobustBench model zoo — no teacher training needed).
- RSLAD (Zi et al., ICCV 2021): RN18 student AA ~51.5 (C10).
- AdaAD (Huang et al., CVPR 2023, github.com/boyellow/AdaAD): reported RN18-student
  AA up to ~52.9 with a WRN-34-20 teacher — the best RN18-class no-extra-data
  numbers we found anywhere.
- IGDM (indirect gradient matching, arXiv:2312.03286, 2024): further +0.5-1 on top
  of distillation methods.
- Feasibility: teacher forward passes only (fits 8GB at eval batch), student =
  RN18. ~1.5x AT cost. Expected with our WSC practices stacked: **AA 52-53.5**.
- This is the most compute-realistic path to clearly beating our 50.49 baseline.

### 3. ReBAT — rebalanced AT against robust overfitting (+0.5 to +0.9, trivial)
NeurIPS 2023 (arXiv:2310.19360, github.com/PKU-ML/ReBAT): re-balance the minimax
game after the LR drop (stronger attack later in training + WA). Reported
PreActRN18 C10 AA **51.13-51.22** — above our 50.49 with no extra cost.
Feasibility: a few-line change to our v3 script + our EMA already half-covers it.

## Tier 2 — moderate, cheap, stackable

### 4. MeanSparse post-processing (+0.3 to +0.7 AA, FREE — no training)
2024 RobustBench entries apply post-hoc mean-centered feature sparsification to
existing checkpoints and gain robustness at ~no clean cost. Apply to our saved
best_ema checkpoints (incl. the 50.78 seed) — could clear 51 without training a
single epoch. Feasibility: an activation-wrapper module + threshold calibration
pass. (Search RobustBench 2024 "MeanSparse" entries for reference implementation.)

### 5. Swish/SiLU activation (+0.3 to +0.5)
Gowal et al. 2020 ablations and all top RobustBench models use Swish instead of
ReLU. One-line change in models/resnet.py (keep ReLU checkpoints separate).

### 6. ADR — annealing self-distillation rectification (+0.4 to +1.0)
ICLR-track 2023 (arXiv:2305.12118): soften/rectify labels with an annealed EMA
teacher; combines with WA+AWP (reported RN18 C10 AA ~50.6-51.1). Stacks naturally
on our recipe (we already have the EMA model to serve as the teacher).

### 7. Architecture tweak at fixed budget: RobustResNet-A1 (+1 to +1.5)
"Revisiting Residual Networks for Adversarial Robustness" (arXiv:2212.11005):
robustness-aware block/width redesign at RN18-like FLOPs beats RN18 by >1 AA.
Drop-in model file; our recipe unchanged. Blurs "same architecture" comparability
with CAP — report as its own row.

### 8. Consistency regularization over augmentations (+0.3 to +0.7)
Tack et al., AAAI 2022 (github.com/alinlab/consistency-adversarial): enforce
consistent predictions across augmented adversarial views. Cheap; complements
our objective; also reported to help common-corruption robustness.

## Tier 3 — our own novel angles (paper-differentiating, unproven)

### 9. Heterogeneous-vertex CHM ("hetero-hull") — direct fix to CHM's measured
failure mode. Our analysis (PAPER_DRAFT.md 4.4) found KL-PGD restarts largely
agree -> hull too thin to bind. Replace same-objective restarts with vertices from
*different* attack losses (CE-PGD, CW-margin-PGD, KL-PGD): systematically diverse
maximizers, fatter hull, max no longer trivially at one vertex. Cost unchanged
(N attacks either way). If the hull idea can win anywhere at eps=8/255, it is
here. Instrument the "fraction of batches where a mix/vertex-2 attains the max"
metric to verify diversity actually increased.

### 10. CHM at larger eps and/or on WRN (where the hull has room to bind)
Vertex diversity grows with the ball radius; inner-max quality matters more with
capacity. Run the eps sweep (10, 12, 16/255) RN18 C10: cheap (same script, one
flag) and gives the paper a "where it wins" curve even if 8/255 stays a tie.

### 11. CHM + distillation hull ("teacher-disagreement vertices")
Combine 2 + 9: vertices = points where student maximally disagrees with the robust
teacher (AdaAD's inner objective) from multiple restarts/losses; hull-max the
disagreement. Novel, plausible, and rides the strongest cheap lever (distillation)
so the headline number lands in the 52-53 range where reviewers pay attention.

## Suggested attack plan (2x RTX 2080, ~2 weeks)
1. **MeanSparse on existing checkpoints** (free, days: 0.5) -> possibly >51 immediately.
2. **ReBAT-ify v3 + Swish** (days: 1.5, both GPUs) -> expect ~51.3-51.8 recipe floor.
3. **AdaAD distillation** with a downloaded RobustBench WRN-34-10 teacher
   (days: 2-3) -> expect 52-53; add IGDM if time.
4. **Hetero-hull CHM** on top of the best of (2)/(3) (days: 2-3, A/B with N=1
   anchor, 2 seeds, pre-committed criterion as always) -> the novelty experiment.
5. **Generated-data track** in parallel whenever GPUs idle (days: 3+ per run) ->
   the 54-57 headline if the paper allows an extra-data track.

## Sources
- Wang et al. 2023, ICML: arXiv:2302.04638; github.com/wzekai99/DM-Improves-AT
- Gowal et al. 2021: arXiv:2110.09468
- Zi et al. 2021 (RSLAD): arXiv:2108.07969
- Huang et al. 2023 (AdaAD): CVPR 2023; github.com/boyellow/AdaAD
- IGDM 2024: arXiv:2312.03286
- ReBAT, NeurIPS 2023: arXiv:2310.19360; github.com/PKU-ML/ReBAT
- ADR: arXiv:2305.12118
- RobustResNets: arXiv:2212.11005
- Consistency reg.: arXiv:2103.04623; github.com/alinlab/consistency-adversarial
- RobustBench: robustbench.github.io (model zoo for teachers; MeanSparse entries)
