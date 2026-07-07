# Project plan: beat CAP (ICASSP 2024) on CIFAR-10 / ResNet-18 / Linf 8/255

Created 2026-06-11 (Claude session 2). Old files untouched; new code lives in
`main_adv_CHM_v3.py`. Running notes continue in `CHANGES.md`.

## 1. The target

CAP = "Robustness Against Adversarial Attacks Via Learning Confined Adversarial
Polytopes", Mohajer Hamidi & Ye, ICASSP 2024 (DOI 10.1109/ICASSP48485.2024.10446776).
Their headline numbers, ResNet-18, CIFAR-10, Linf eps=8/255, 120 epochs
(SGD 0.1, /10 at epochs 80 and 100, bs 128, wd 5e-4):

| Method | Clean | FGSM | PGD-20 | PGD-100 | C&W | AA |
|---|---|---|---|---|---|---|
| Vanilla AT | 82.78 | 56.94 | 51.30 | 50.88 | 49.72 | 47.63 |
| TRADES (β=6) | 82.41 | 58.47 | 52.76 | 52.47 | 50.43 | 49.37 |
| MART | 80.70 | 58.91 | 54.02 | 53.58 | 49.35 | 47.49 |
| **CAP (λ=0.6, N=10, T=40)** | **83.04** | **59.23** | **54.31** | **54.09** | **50.85** | **50.24** |

What CAP actually is: TRADES-shaped loss where the regularizer is
`λ Σ_n ||f(x+ε*_n) − C*_x||²` — N=10 "particles" pushed (T=40 PGD steps!) toward the
*corners* of the output polytope, then pulled back to the polytope center. Two
observations that define our attack surface:

1. CAP's gain over TRADES is ≈ +0.9 AA. That is *smaller* than the known gains from
   AWP (+1.2 AA over TRADES) or good augmentation + weight averaging (+1–2 AA), which
   CAP does not use. Their recipe is otherwise vanilla.
2. CAP's particle search costs N=10 × T=40 = 400 forward/backward per batch (they
   parallelize over N, still 40 sequential steps). We can spend far less compute on a
   better-understood regularizer and win on recipe quality.

## 2. Where the points come from (literature, ResNet-18, no extra data)

| Lever | Expected effect (AA) | Reference |
|---|---|---|
| TRADES β=6, 10-step KL attack | baseline ≈49.4 | Zhang et al. 2019, arXiv:1901.08573 |
| + AWP (γ=5e-3) | ≈50.6 (+1.2) | Wu et al., NeurIPS 2020, arXiv:2004.05884; github.com/csdongxian/AWP (TRADES-AWP RN18 AA 50.58) |
| + weight averaging (EMA) | +0.5–1.0, also +clean | Gowal et al. 2020, arXiv:2010.03593; Izmailov et al. 2018, arXiv:1803.05407 |
| + stronger augmentation (Cutout/RE; IDBH-style) | +0.5–1.0 with WA | Li & Spratling, ICLR 2023 "Data augmentation alone can improve adversarial robustness" (IDBH); Rebuffi et al. 2021, arXiv:2103.01946 |
| robust-val model selection (early-stop) | protects all of the above | Rice et al. 2020, arXiv:2002.11569 |
| + CHM hull term (ours) | +clean at iso-robustness (measured v2: +2.1 clean), possibly +robust now that it rides a stronger base | this repo, CHANGES.md session 1 |

Related recipes confirming ~50.5–51 AA is the RN18 no-extra-data band: HAT (Rade &
Moosavi-Dezfooli, ICLR 2022, github.com/imrahulr/hat, AA 50.62 with clean 84.86);
ADR+WA+AWP (Wu et al. 2023, arXiv:2305.12118, AA 50.59). Going meaningfully above
~52 on RN18 without generated/extra data is beyond published results — so the win
condition is: **AA > 50.24 and PGD-20 > 54.31 with clean ≥ 83**, plus a better
clean/robust trade-off story from CHM.

## 3. Our method for the paper: CHM-TRADES ("worst-case-over-hull TRADES")

Loss per example:

  CE(f(x), y) + β · max_{δ ∈ Hull} KL( f(x+δ) ‖ f(x) )

where Hull = convex hull of {δ_1..δ_N} TRADES-KL-PGD solutions from independent
random starts, approximated by the N vertices plus K Dirichlet convex combinations
(interior points). With N=1, K=0 this is *exactly* TRADES — clean ablation story.

Positioning vs CAP (the paper's pitch):
- CAP confines the *output* polytope by pushing sampled corners toward the center —
  a symmetric, attack-agnostic compactness prior in logit space.
- CHM-TRADES instead minimizes the *worst-case divergence over the input-space hull*
  of strong perturbations: a direct surrogate of the robust risk, asymmetric (only
  the worst point matters), and 20× cheaper than CAP's particle search
  (N=2 restarts × 10 steps vs N=10 × 40 steps).
- Dirichlet interior sampling is valid because the Linf ball intersected with the
  [0,1] box is convex (mixes of feasible deltas are feasible).

## 4. Run matrix (2× RTX 2080 8GB)

All runs: ResNet-18, bs 128, SGD 0.1/0.9/5e-4, 120 epochs, /10 @ 80 & 100 (matching
CAP), train attack = TRADES-KL PGD-10 step 2/255, EMA decay 0.995 tracked alongside,
model selection by PGD-10 robust acc on held-out 1k train split, last+best saved.

| run | GPU | config | role |
|---|---|---|---|
| v3_trades_awp_base | 0 | N_hull=1, K_mix=0, β=6, AWP γ=5e-3 | reproduce TRADES-AWP ≈50.6 AA; our "recipe floor" |
| v3_chm_trades_awp | 1 | N_hull=2, K_mix=2, β=6, AWP γ=5e-3 | our method on the same recipe |

Phase 2 (after first results): seeds ×2, β/λ sweep if CHM term underweighted,
Cutout/RE augmentation arm, optional WRN-34-10 if RN18 wins (8GB may force bs 64 +
grad accum).

Evaluation protocol (decided up front to avoid cherry-picking):
- During training: PGD-10 val (selection only), PGD-20 test every 10 epochs (monitor).
- Final: full-test PGD-20 + AA-fast (APGD-CE+T, 1000) on best_net/best_ema of both
  arms; then **full standard AutoAttack (10k)** only on each arm's chosen checkpoint.
- Report mean of ≥2 seeds before writing the paper numbers.

## 5. Risks / honesty notes

- v2 showed the hull term ≈ tie on robustness; the paper's fallback claim is the
  clean/robust trade-off (+2.1 clean at iso-robust). The v3 base is much stronger, so
  re-measure rather than assume.
- 8 GB cards: AWP adds a full proxy model copy; batch 128 RN18 fits (~4 GB), but the
  CHM arm does N+K+1 forwards per step — verified by smoke test before launch.
- AA-fast on 1000 examples has ±1.5 noise; never claim a win from it. Full AA 10k
  only for the final table.
- Compute per epoch: CHM arm ≈2× baseline arm (20 vs 10 attack steps + 5 vs 2 loss
  forwards). At ~8–9 min/epoch worst case → 120 epochs ≈ 17 h on the 2080. Baseline
  ≈ 8–9 h.
