# Experiment log & gap analysis vs CAP (compiled 2026-07-07)

All experiments run on 2x RTX 2080 (8GB), conda env `hullrob`, this repo.
Full narrative/rationale in CHANGES.md; paper framing in paper/PAPER_DRAFT.md.
CAP reference: Mohajer Hamidi & Ye, ICASSP 2024, arXiv:2401.07991 (v2 HTML).

Threat model everywhere: Linf, eps = 8/255. Attacks at eval unless noted:
PGD-20 (alpha=2/255, random start, full 10k test set) and AutoAttack
(fra31 standard version: APGD-CE, APGD-T, FAB-T, Square; full 10k unless noted).
"AA-fast" = APGD-CE+APGD-T on 1000 examples — monitoring only, never for claims.

## A. Everything we ran

### A1. Evaluation of pre-existing checkpoints (v1-era, 50ep cosine, PGD-AT + hull margin)

| checkpoint | Clean | PGD-20 | notes |
|---|---|---|---|
| res18_steps5_Nh2_lam0.0 | 86.27 | 44.93 | baseline; AA-fast(1k) 42.60 |
| res18_steps5_Nh2_lam0.005 | 86.35 | 44.48 | CHM, matched steps -> no gain |
| res18_steps6_Nh2_lam0.002 | 85.75 | 46.15 | |
| res18_steps6_Nh2_lam0.005 | 85.66 | 46.95 | AA-fast(1k) 44.60 |
| res18_steps6_Nh2_lam0.01 | 85.58 | 46.20 | |
| res18_steps6_Nh4_lam0.005 | 85.59 | 46.15 | N_hull 4 = 2 -> variants wasted |
| res18_steps7_Nh2_lam0.005 | 84.98 | 47.51 | gains track PGD steps, not hull |

(Original AutoAttack eval was invalid — normalized inputs fed to a [0,1] library;
rewritten with a normalization wrapper. The old "26.07%" is void.)

### A2. v2 A/B (50 epochs, cosine, PGD-AT, worst-case CE over restarts, EMA 0.999,
robust-val selection; seed: default)

| arm | ckpt | Clean | PGD-20 | AA-fast(1k) |
|---|---|---|---|---|
| PGD-AT baseline (Nh1 lam0) | best_net | 81.07 | 48.60 | 46.40 |
| PGD-AT baseline | best_ema | 84.31 | 47.59 | 44.00 |
| CHM (Nh2 lam0.005 both K2) | best_net | 83.21 | 48.83 | 45.70 |
| CHM | best_ema | 78.32 | 48.26 | 44.60 |

Finding: recipe fixes worth +3.7 PGD-20 over v1; CHM = robustness tie, +2.1 clean
(did NOT replicate on the stronger v3 recipe -> headroom artifact).

### A3. v3 CIFAR-10 main arms (CAP recipe: 120ep, SGD 0.1 /10 @80,100, bs128,
wd 5e-4; TRADES beta=6 KL-PGD-10; AWP gamma=5e-3; EMA 0.995; robust-val selection)

Full AA = standard 10k. best_ema unless noted.

| run | seed | Clean | PGD-20 | full AA |
|---|---|---|---|---|
| v3_trades_awp_base_s0 (control) | 0 | 81.15 | 54.63 | 50.78 |
| v3_chm_trades_awp_s0 (N=2,K=2) | 0 | 79.89 | 54.14 | 50.11 |
| v3_chm_trades_awp_s1 | 1 | 80.16 | 54.18 | 49.95 |

### A4. v3 CIFAR-10 recipe sweeps (single seed each, negative/lever results)

| run | Clean | PGD-20 | AA (type) |
|---|---|---|---|
| b4+RE (beta=4, RandErase .5) best_ema | 82.47 | 53.39 | 48.60 (fast 1k) |
| b5+RE (beta=5, RandErase .5) best_ema | 81.82 | 53.50 | 48.60 (fast 1k) |
| g3 (awp 3e-3) best_ema | 81.48 | 54.47 | 49.00 (fast 1k) |
| g2 (awp 2e-3) s0 best_net | 82.15 | 54.07 | 50.27 (full 10k) |
| g2 (awp 2e-3) s0 best_ema | 82.30 | 54.13 | 50.74 (full 10k) |
| g2 (awp 2e-3) s1 best_ema | 81.66 | 54.24 | 50.24 (full 10k) |

Findings: RE+lower beta = strictly inside the frontier (discarded). AWP gamma =
the clean<->robust lever. g2 2-seed mean: 81.98 / 54.19 / AA 50.49 +/- 0.25.

### A5. CIFAR-100 A/B (same v3 recipe, gamma=5e-3; 2 seeds; full AA on ALL ckpts)

| arm | seed | ckpt | Clean | PGD-20 | full AA |
|---|---|---|---|---|---|
| control | 0 | best_net* | 57.60 | 30.48 | 25.09 |
| control | 0 | best_ema | 58.05 | 30.90 | 25.71 |
| control | 1 | best_net* | 58.63 | 31.08 | 25.87 |
| control | 1 | best_ema | 58.82 | 31.12 | 25.71 |
| CHM | 0 | best_net* | 57.51 | 30.90 | 25.61 |
| CHM | 0 | best_ema | 57.91 | 30.84 | 25.38 |
| CHM | 1 | best_net | 57.53 | 30.58 | 25.44 |
| CHM | 1 | best_ema* | 57.67 | 31.01 | 25.52 |

(* = val-chosen.) 2-seed val-chosen means: control 58.12 / 30.78 / 25.48+/-0.39;
CHM 57.59 / 30.96 / 25.57+/-0.05. Tie.

### A6. Headline results

| | Clean | PGD-20 | full AA |
|---|---|---|---|
| Ours C10 (TRADES-AWP g2, 2 seeds) | 81.98 | 54.19 | 50.49 +/- 0.25 |
| CAP C10 (5 runs) | 83.04 +/- .13 | 54.31 +/- .10 | 50.24 +/- .10 |
| Ours C100 (control, 2 seeds) | 58.12 | 30.78 | 25.48 +/- 0.39 |
| CAP C100 (5 runs) | 58.02 +/- .17 | 30.44 +/- .15 | 25.42 +/- .09 |

CHM (the novel term): negative result, 2 seeds x 2 datasets (loss on C10, tie on C100,
2x cost).

## B. CAP's experimental suite (arXiv:2401.07991) vs ours — GAP ANALYSIS

CAP evaluated: CIFAR-10 + CIFAR-100 + SVHN; ResNet-18 + WRN-34-5 + WRN-34-10;
attacks Clean / FGSM / PGD-20 / PGD-100 / C&W-inf / AA (eps 8/255); 5 runs with
95% CIs; baselines Vanilla AT, TRADES (beta 6), MART (beta 5). SVHN recipe differs:
80ep, lr 0.01 /10 @50,65, alpha=1/255. No ablations reported.

| CAP has | We have | Gap | Cost to close (2x RTX 2080) |
|---|---|---|---|
| PGD-20, AA | same (AA full 10k) | none | — |
| FGSM eval | NOT RUN | add FGSM to eval_checkpoint.py (torchattacks.FGSM) | minutes/ckpt — trivial |
| PGD-100 eval | NOT RUN | add --pgd_steps 100 pass | ~15 min/ckpt |
| C&W-inf eval | NOT RUN | PGD-100 on CW margin loss (their C&W-inf is PGD-style on CW loss); small script addition | ~15 min/ckpt |
| 5 seeds, 95% CI | 2 seeds (main arms), 1 seed (sweeps) | 3 more seeds per headline arm | ~9h/run C10, ~9-17h C100 -> ~5-6 GPU-days for both arms x both datasets |
| SVHN (80ep, lr .01, /10 @50,65, alpha 1/255) | NOT RUN | --dataset svhn + their recipe params in main_adv_CHM_v3.py (torchvision SVHN; no 50k assumption — 73,257 train) | ~1 day/arm-pair incl. evals |
| WRN-34-5 / WRN-34-10 | NOT RUN (RN18 only) | add WRN model; 8GB VRAM likely forces bs 64 + grad accum; slower epochs | multi-day/run — the expensive gap |
| Vanilla AT baseline (their recipe, 120ep) | v2 50ep cosine only — NOT comparable | rerun vanilla AT at CAP recipe if we want their Table-1 rows reproduced | ~7h/seed |
| TRADES (no AWP) baseline | NOT RUN (we always used +AWP, deliberately stronger) | plain TRADES beta=6 at 120ep would let us reproduce their 49.37 row | ~7h/seed |
| MART baseline | NOT RUN | implement MART loss (arXiv 1910.08051-style) | ~1 day incl. impl |
| CAP itself (reimplementation) | NOT RUN (numbers quoted from their paper) | N=10 x T=40 particle search -> ~20x regularizer cost; days/run on 2080s | expensive; paper currently quotes their numbers with an explicit caption |

### Notes on protocol differences (ours, intentional — not gaps)
- We select checkpoints by robust val accuracy on a held-out 1k train split
  (Rice et al.); CAP does not report a selection rule. Ours is the stricter protocol.
- We report full-10k standard AutoAttack for every claim; monitoring-only AA-fast
  is never quoted.
- Our TRADES arm includes AWP+EMA (deliberately stronger control for the CHM
  question); it is NOT a reproduction of their TRADES row.

### Priority order to reach CAP-table parity (if desired)
1. FGSM / PGD-100 / C&W-inf on the 4 headline checkpoints — <1 day total, completes
   the attack columns on both datasets.
2. Seeds 2-4 for the two headline arms on C10+C100 — brings 5-seed CIs, ~5-6 GPU-days.
3. SVHN arm-pair — 1-2 days including --dataset svhn support.
4. WRN-34-10 arm-pair — the big one; only worth it if the paper targets a venue
   where RN18-only is a likely reject.
5. Own reproductions of vanilla AT / TRADES / MART / CAP — only needed if reviewers
   reject quoted-numbers comparison.

## A7. Full CAP attack battery (added 2026-07-07, eval_cap_suite.py)

Clean/FGSM/PGD-20/PGD-100/CW-inf from eval_cap_suite.py ([0,1]-space, alpha=2/255,
CW-inf = PGD-100 on the CW margin loss); AA = full-10k standard from eval_autoattack.py.
Per-checkpoint values in output/cap_suite_c10.log, output/cap_suite_c100.log.

### CIFAR-10 (2-seed means; CAP rows = their 5-run means)

| method | Clean | FGSM | PGD-20 | PGD-100 | CW-inf | AA |
|---|---|---|---|---|---|---|
| Ours TRADES-AWP (g2) | 81.98 | 58.28 | 54.14 | 54.00 | **51.24** | **50.49** |
| Ours CHM | 80.03 | 57.30 | 54.19 | 54.08 | 50.51 | 50.03 |
| CAP (paper) | 83.04 | 59.23 | 54.31 | 54.09 | 50.85 | 50.24 |
| TRADES (CAP's table) | 82.41 | 58.47 | 52.76 | 52.47 | 50.43 | 49.37 |

Reading: ties CAP on PGD-20/PGD-100/AA within seed noise, BEATS CAP on CW-inf
(+0.39), loses clean (-1.06) and FGSM (-0.95; weakest attack, tracks clean).

### CIFAR-100 (2-seed means)

| method | Clean | FGSM | PGD-20 | PGD-100 | CW-inf | AA |
|---|---|---|---|---|---|---|
| Ours TRADES-AWP (control) | 58.12 | 33.26 | **30.84** | **30.72** | 26.40 | 25.48 |
| Ours CHM | 57.59 | 32.97 | 30.88 | 30.87 | 26.49 | 25.57 |
| CAP (paper) | 58.02 | 33.27 | 30.44 | 30.27 | 26.66 | 25.42 |

Reading: on CIFAR-100 the tuned TRADES-AWP control reaches FULL PARITY OR BETTER
with CAP on every column — clean +0.10, FGSM -0.01, PGD-20 +0.40, PGD-100 +0.45,
CW-inf -0.26, AA +0.06 — at ~1/20 the regularizer cost. The clean-accuracy deficit
is a CIFAR-10-only phenomenon.

Remaining gaps vs CAP after this battery: seeds 3-5, SVHN, WRN-34-5/10,
own baseline reproductions (see section B priorities).
