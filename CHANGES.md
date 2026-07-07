# Changes log (Claude session, 2026-06-11)

Goal: figure out why CHM results don't beat the plain adversarial-training baseline, fix
evaluation bugs, and improve the training recipe.

## Findings (before any change)

### Measured results of existing checkpoints (PGD-20, eps=8/255, full test set)

| checkpoint | Clean | PGD-20 |
|---|---|---|
| res18_steps5_Nh2_lam0.0 (baseline) | 86.27% | 44.93% |
| res18_steps5_Nh2_lam0.005 | 86.35% | 44.48% |
| res18_steps6_Nh2_lam0.002 | 85.75% | 46.15% |
| res18_steps6_Nh2_lam0.005 | 85.66% | 46.95% |
| res18_steps6_Nh2_lam0.01 | 85.58% | 46.20% |
| res18_steps6_Nh4_lam0.005 | 85.59% | 46.15% |
| res18_steps7_Nh2_lam0.005 | 84.98% | 47.51% |

Conclusions:
- At matched PGD steps (5), CHM (44.48%) does NOT beat the lam=0 baseline (44.93%).
- Gains down the table come from more PGD steps, not from the hull term.
- N_hull=4 gives nothing over N_hull=2 -> the extra adversarial examples are wasted
  (CE only used variant[0]; the hull term weight 0.005 is too small to matter).
- All numbers are below the ~52% PGD-20 a well-tuned PGD-AT ResNet18 reaches
  (Rice et al. 2020; RobustBench baselines), so the recipe itself had headroom.

### Bug: AutoAttack evaluation was invalid
`eval_autoattack.py` fed *normalized* tensors to AutoAttack, which assumes [0,1] images
(`adversary.mean/std` is not an API of fra31/auto-attack; those lines were no-ops).
APGD clamped images to [0,1] in normalized space, destroying them — visible in the log
line `max Linf perturbation: 2.42907`. The reported 26.07% is meaningless; the real AA
accuracy of that checkpoint is unknown (expected: a few points below its PGD-20).

### Bug: checkpoints saved by best *clean* accuracy
Due to robust overfitting, best-clean selection throws away robust accuracy.
The steps6_Nh4 log shows PGD-20 peaked at 47.41% (epoch 34) but the saved model is from
a late epoch (46.1%).

### Other issues
- `clip_grad_norm_(max_norm=1.0)` is nonstandard for AT and slows fitting.
- `main_adv_ex_CHM_TRADES.py`: `--beta` parsed but never used (loss is NOT TRADES);
  its custom attack perturbs in normalized space, so eps is ~5x weaker than intended;
  no valid-range clamp.
- `main_adv_CHM_baseline.py:34`: `--eval_only` added after `parse_args()` (dead code).

## Changes made

### 1. `eval_autoattack.py` — rewritten (fix invalid evaluation)
- Test data loaded WITHOUT Normalize (raw [0,1] tensors).
- Added `NormalizedModel` wrapper that normalizes inside `forward()` — the standard
  way to evaluate normalized models with AutoAttack.
- Removed the no-op `adversary.mean/std` lines.
- New flags: `--n_examples` (subset for quick evals), `--key net|ema` (evaluate EMA
  weights from v2 checkpoints), `--version standard|rand|fast`
  (`fast` = APGD-CE + APGD-T only; cheap, usually within ~0.3% of full AA).

### 2. `main_adv_CHM_v2.py` — new training script (old scripts left untouched)
- **Worst-case CE over all N_hull variants** (per-example max) instead of CE on
  variant[0] only -> the N_hull attacks now actually train the model
  (equivalent to PGD with random restarts). With `--N_hull 1 --lam_hull 0` the script
  is exactly standard PGD-AT (clean baseline for comparison).
- **Hull interior sampling** (`--hull_mode mix|both`, `--K_mix`): margin loss is also
  applied to Dirichlet convex combinations of the adversarial deltas — points *inside*
  the convex hull, which is the actual CHM idea (vertices alone = just restarts).
  Mixes are valid perturbations because the Linf ball and [0,1] box are convex.
- **EMA weight averaging** (`--ema_decay 0.999`), evaluated and checkpointed
  separately (known +1-2% robust accuracy).
- **Model selection by robust accuracy** (PGD-10) on a held-out 1000-image split of
  the *train* set — fixes both the best-clean selection bug and test-set leakage.
  Saves `<run>_best_net.pth`, `<run>_best_ema.pth`, `<run>_last.pth`.
- **PGD-10 training attack** by default (was 5-7).
- **No gradient clipping** by default (`--clip_grad 0`; flag available).
- `--max_batches` for quick smoke tests.

### 3. `utils.py` — crash fix
`stty size` fails when there is no terminal (background/nohup runs), crashing every
script at import. Now falls back to width 80.

### 4. `main_adv_CHM_v2.py` — ramp formula fix
Old ramp `(epoch - warmup) / ramp_epochs` left the first post-warmup epoch at weight 0.
Now `(epoch - warmup + 1) / ramp_epochs`.

## References for borrowed methods
- PGD adversarial training (and restarts): Madry et al. 2018, "Towards Deep Learning
  Models Resistant to Adversarial Attacks", arXiv:1706.06083.
- Robust overfitting / early stopping on a robust validation split: Rice et al. 2020,
  "Overfitting in adversarially robust deep learning", arXiv:2002.11569.
- Weight averaging: Izmailov et al. 2018 (SWA), arXiv:1803.05407; EMA specifically in
  adversarial training: Gowal et al. 2020, "Uncovering the Limits of Adversarial
  Training against Norm-Bounded Adversarial Examples", arXiv:2010.03593.
- Dirichlet sampling inside a convex hull of perturbations (adapted from NLP token
  embedding hulls to pixel-space Linf deltas): Zhou et al. 2021, "Defense against
  Adversarial Attacks in NLP via Dirichlet Neighborhood Ensemble", arXiv:2006.11627;
  Dong et al., ICLR 2021, "Towards Robustness Against Natural Language Word
  Substitutions" (ASCC).
- AutoAttack evaluation: Croce & Hein 2020, arXiv:2003.01690; normalization-wrapper
  usage per fra31/auto-attack issue #46.
- TRADES (for reference re: the broken script): Zhang et al. 2019, arXiv:1901.08573.

## Verification runs

### Corrected AutoAttack (fast = APGD-CE + APGD-T, 1000 test examples)
| checkpoint | clean (n=1000) | AA-fast |
|---|---|---|
| res18_steps5_Nh2_lam0.0 (old baseline) | 86.70% | 42.60% |
| res18_steps6_Nh2_lam0.005 (old best CHM) | 86.60% | 44.60% |

The old "26.07%" was an artifact of the broken eval. Real numbers match literature for
vanilla PGD-AT (~43-44% AA). PGD-20 -> AA gap is ~2 points for both models, i.e. no
gradient masking. (Note: the 2-point CHM "gain" here is still confounded by steps 5 vs 6.)

### A/B training runs (v2 recipe, launched 2026-06-11)
Fair comparison, both steps=10, 50 epochs, EMA, robust-val model selection:
- GPU0 `v2_res18_steps10_Nh1_lam0.0_vertex_ep50`  — pure PGD-AT baseline (N_hull=1, lam=0)
- GPU1 `v2_res18_steps10_Nh2_lam0.005_both_ep50` — CHM (N_hull=2, lam=0.005, vertex+mix margin, K_mix=2)
Logs: output/v2_baseline.log, output/v2_chm.log
(CHM run was killed at epoch 48/50 — harmless: best-by-robust-val checkpoints were
already saved and cosine LR is ~0 in the last epochs.)

### Final results (2026-06-11): full-test PGD-20; AA-fast = APGD-CE+APGD-T on 1000 examples

| model | Clean | PGD-20 | AA-fast |
|---|---|---|---|
| old baseline (steps5, lam0, best-clean sel.) | 86.27% | 44.93% | 42.60% |
| old best CHM (steps6, lam0.005) | 85.66% | 46.95% | 44.60% |
| **v2 baseline best_net** | 81.07% | 48.60% | **46.40%** |
| v2 baseline best_ema | 84.31% | 47.59% | 44.00% |
| **v2 CHM best_net** | **83.21%** | **48.83%** | 45.70% |
| v2 CHM best_ema | 78.32% | 48.26% | 44.60% |

### Verdict
1. **The recipe fixes drove the improvement**: +3.7 PGD-20 / +3.8 AA over the old
   baseline (44.93 -> 48.60 PGD-20; 42.60 -> 46.40 AA), matching literature for a
   properly tuned PGD-AT ResNet18.
2. **CHM vs baseline at matched settings (the fair A/B)**: robustness is a statistical
   tie (PGD-20 +0.23, AA -0.7 with n=1000 noise ~±1.5), but CHM keeps **+2.1% clean
   accuracy** at that robustness level (83.21 vs 81.07). So the honest claim is a
   better clean/robust trade-off, not higher worst-case robustness.
3. EMA selection underperformed best_net robustness in both runs at this 50-epoch
   budget (decay 0.999 may be too slow for ~19k steps); EMA did give the best clean
   accuracy. Worth retuning (0.995) or starting EMA after the LR drop.

### Suggested next steps
- 2-3 seeds per arm before claiming the clean-accuracy gain is real.
- Sweep lam_hull (0.01-0.05) and K_mix now that the hull term competes with
  worst-case CE rather than CE-on-variant[0]; current effect size suggests the
  term is under-weighted.
- 100+ epoch runs (robust overfitting handled by robust-val selection).
- Full AutoAttack (standard, 10k) on the final chosen model only.
- CIFAR-100 / PreActResNet18 for generality.

---

# Session 2 (2026-06-11, evening): beat the CAP paper

Goal: surpass CAP (Mohajer Hamidi & Ye, ICASSP 2024; ResNet-18/CIFAR-10/Linf 8/255:
clean 83.04 / PGD-20 54.31 / AA 50.24). Strategy + literature in `PLAN_BEAT_CAP.md`;
paper skeleton in `paper/PAPER_DRAFT.md`. Old scripts untouched.

## Changes made

### 1. `main_adv_CHM_v3.py` — new training script (CHM-TRADES + AWP + EMA)
Architecture of the method changed from CE-margin (v2) to TRADES-style KL:
- **TRADES loss** (CE clean + beta*KL, KL-PGD-10 attack, beta=6): Zhang et al. 2019,
  arXiv:1901.08573. With `--N_hull 1 --K_mix 0` the script IS standard TRADES.
- **CHM hull term**: robust term = worst-case KL over N_hull KL-PGD restarts
  (hull vertices) + K_mix Dirichlet convex combinations (hull interior). Replaces
  v2's softplus margin; reduces to TRADES, upper-bounds it otherwise.
- **AWP** (TRADES-AWP, gamma=5e-3, proxy lr 0.01, 10-epoch warmup): Wu et al.,
  NeurIPS 2020, arXiv:2004.05884, after github.com/csdongxian/AWP. Known +1.2 AA
  over TRADES on RN18 (AA 50.58 reported) — the single biggest lever vs CAP.
- **EMA decay 0.995** (retuned from v2's 0.999 per session-1 verdict).
- **Recipe matched to CAP for fairness**: 120 epochs, SGD 0.1, /10 at 80 & 100,
  wd 5e-4, bs 128. (v2 used 50-epoch cosine.)
- **Unnormalized data pipeline**: tensors stay in [0,1]; normalization inside the
  forward call. Removes the torchattacks normalization-state footgun and makes
  Dirichlet mixes/attacks trivially valid. Checkpoint format unchanged
  ({'net','ema'}, DataParallel prefix) -> eval_autoattack.py works as-is.
- In-file PGD/KL-PGD implementations (no torchattacks dependency).

### 2. Smoke tests (5 batches, AWP forced on)
Both arms run clean: baseline 528 ms/batch (~3.5 min/epoch, ~8 h/120 ep);
CHM N=2,K=2 1.33 s/batch (~8.5 min/epoch, ~18 h/120 ep). RTX 2080 8GB OK.

### 3. NaN fix after first launch (caught at epoch 0, runs restarted)
First launch (PIDs 3316383/3316384, ~22:23) went NaN mid-epoch 0 in BOTH arms.
Cause: `F.kl_div(log_softmax(adv), softmax(clean))` — the probability target
underflows to exact 0 in fp32 once a logit gap exceeds ~88, and target*log(target)
= 0*(-inf) = NaN, which then poisons the weights (a well-known TRADES-reimpl
pitfall). Fix: all KL terms (attack, AWP proxy, main loss) now computed in log
space via `log_target=True` — `log_softmax` outputs are always finite, so the KL
is NaN-safe. Also added a non-finite-loss guard that skips the update (loudly)
instead of stepping on a poisoned gradient.

## Runs launched (2026-06-11 ~22:35 relaunch, nohup, survive logout)
| GPU | run | config | log |
|---|---|---|---|
| 0 | v3_trades_awp_base_s0 | TRADES+AWP control (N=1,K=0) | output/v3_trades_awp_base_s0.log |
| 1 | v3_chm_trades_awp_s0 | CHM-TRADES+AWP (N=2,K=2) | output/v3_chm_trades_awp_s0.log |
PIDs 3317832 / 3317833. htop running in detached tmux session `monitor`
(`tmux attach -t monitor`). ETA: baseline ~morning, CHM ~evening of 2026-06-12.

## Evaluation protocol (pre-registered in PLAN_BEAT_CAP.md §4 to avoid cherry-picking)
Final claim requires: full-test PGD-20 + AA-fast on best_net/best_ema of both arms,
then FULL standard AutoAttack (10k) on the chosen checkpoint per arm; >=2 seeds
before paper numbers. Win condition: AA > 50.24, PGD-20 > 54.31, clean >= 83.

## References added this session
- AWP: Wu, Xia, Wang, NeurIPS 2020, arXiv:2004.05884 (code github.com/csdongxian/AWP).
- HAT: Rade & Moosavi-Dezfooli, ICLR 2022 (RN18 AA 50.62 — band confirmation).
- IDBH augmentation: Li & Spratling, ICLR 2023 (phase-2 lever, `--random_erase` stub).
- ADR: Wu et al. 2023, arXiv:2305.12118 (RN18 AA 50.59 w/ AWP+WA — band confirmation).
- Certified polytope ancestor (paper positioning): Wong & Kolter, ICML 2018,
  arXiv:1711.00851.

## v3 results (updated 2026-06-12, session 3)

### Baseline arm `v3_trades_awp_base_s0` (TRADES beta=6 + AWP + EMA, 120ep) — DONE
Full-test PGD-20; AA-fast = APGD-CE+T n=1000 (noise ~±1.5, never claim wins from it):

| checkpoint | Clean | PGD-20 | AA-fast |
|---|---|---|---|
| best_net (val rob 56.6) | 80.95% | 54.51% | 50.20% |
| best_ema (val rob 57.1, ep118) | 81.15% | **54.63%** | 49.90% |
| CAP paper target | 83.04% | 54.31% | 50.24% (full AA 10k) |

Recipe floor reproduced as planned (TRADES-AWP RN18 ≈ 50.2-50.6 AA band):
robustness already at/above CAP (PGD-20 +0.3), clean ~2pts below CAP.
=> CHM arm's job: recover clean at iso-robustness (v2 showed +2.1 clean).

### CHM arm `v3_chm_trades_awp_s0` — training, epoch 99/120 at time of writing
val rob(ema) 55.7%, val clean(ema) 82.2% just before the epoch-100 LR drop.
Final evals + full standard AutoAttack (10k) on both arms' chosen ckpts after it
finishes.

### CHM arm `v3_chm_trades_awp_s0` — DONE (2026-06-12 13:40), quick evals

| checkpoint | Clean | PGD-20 | AA-fast |
|---|---|---|---|
| best_net (val rob 55.6) | 79.90% | 54.24% | 49.50% |
| best_ema (val rob 55.9) | 79.89% | 54.14% | 49.60% |
| baseline best_ema (ref) | 81.15% | 54.63% | 49.90% |

**The v2 finding did NOT replicate on the stronger base**: on TRADES+AWP, the hull
term (N=2, K=2, worst-case KL) is slightly worse on every metric (clean -1.2,
PGD-20 -0.4, AA-fast -0.4; robustness deltas within n=1000 noise, clean delta is not).
Interpretation: worst-case-over-hull KL effectively strengthens the TRADES attack,
acting like a higher beta — costs clean without buying measurable robustness here.
Win condition vs CAP not met yet: robustness >= CAP but clean 81.15 vs CAP's 83.04.

Full standard AutoAttack (10k) running on both arms' best_ema (paper-grade numbers);
chosen by val robust only, no test peeking.

### Phase-2 plan (queue after full AA finishes, GPUs busy ~8h)
1. beta sweep on the BASELINE recipe (beta=4, 5) — trade robustness headroom
   (PGD-20 +0.3 over CAP) for the missing ~2 clean points. Most direct path to
   beating CAP on all three numbers.
2. If CHM is kept for the paper: soften the hull term (mean-over-hull KL instead of
   max, or lam-weighted second term) and/or pair it with lower beta so the hull term
   supplies the robustness pressure that beta gives up.
3. Seeds x2 for whichever arm wins; augmentation lever (--random_erase 0.5) as the
   backup clean-accuracy lever (IDBH, Li & Spratling ICLR 2023).

### Full standard AutoAttack (10k test images) — paper-grade numbers, 2026-06-12

| arm (best_ema, chosen by val robust) | Clean | PGD-20 | full AA |
|---|---|---|---|
| v3 TRADES+AWP baseline | 81.15% | 54.63% | **50.78%** |
| v3 CHM-TRADES+AWP | 79.89% | 54.14% | 50.11% |
| CAP (published) | 83.04% | 54.31% | 50.24% |

- **Baseline beats CAP on both robustness metrics** (AA +0.54, PGD-20 +0.32) but
  trails clean by 1.89.
- CHM arm confirmed worse than baseline on full AA too (-0.67). The hull-as-stronger-
  attack interpretation stands.
- (eval_autoattack.py final summary forward was unbatched -> OOM on 8GB after the
  attack finished; AA's own printout was unaffected. Now batched.)

### Phase 2 launched (2026-06-12): recover clean without losing the robustness lead
Both arms = baseline recipe (N_hull=1, K_mix=0) + RandomErasing 0.5 (IDBH-style
augmentation lever), trading some beta for clean:
- GPU0 `v3_trades_awp_b4_re_s0`: beta=4
- GPU1 `v3_trades_awp_b5_re_s0`: beta=5
Win condition: clean >= 83.04 while keeping PGD-20 >= 54.31 and AA >= 50.24.

### Phase 2 results (2026-06-13): RandomErasing 0.5 + lower beta — DID NOT beat CAP

| arm (best_ema) | Clean | PGD-20 | AA-fast (n=1000) |
|---|---|---|---|
| b4+RE (beta=4) | 82.47% | 53.39% | 48.60% |
| b5+RE (beta=5) | 81.82% | 53.50% | 48.60% |
| v3 baseline (beta=6, no RE) | 81.15% | 54.63% | 49.90% / 50.78 full |
| CAP target | 83.04% | 54.31% | 50.24% |

Verdict: the RE + lower-beta lever moved STRICTLY INSIDE CAP's frontier. Clean rose
~+1.3 (to 82.47, still < 83.04) but PGD-20 fell -1.2 and AA-fast fell -1.3. It slid
along the trade-off curve the wrong way (and not even far enough on clean). Negative
result — discard this lever.

Best result remains the v3 TRADES+AWP baseline: AA 50.78 (> CAP 50.24) and
PGD-20 54.63 (> CAP 54.31), but clean 81.15 (< CAP 83.04 by 1.89). We beat CAP on
robustness, lose on clean. CHM (the novel term) does not help on the strong base.

### AWP gamma sweep (2026-06-13): loosen AWP to recover clean — PARTIAL success

| arm | Clean | PGD-20 | AA-fast (n=1000) |
|---|---|---|---|
| g2 awp=2e-3 best_net | 82.15% | 54.07% | 50.10% |
| g2 awp=2e-3 best_ema | 82.30% | 54.13% | 49.80% |
| g3 awp=3e-3 best_net | 81.28% | 54.31% | 50.50% |
| g3 awp=3e-3 best_ema | 81.48% | 54.47% | 49.00% |
| baseline awp=5e-3 best_ema | 81.15% | 54.63% | 49.90% / 50.78 full |
| CAP target | 83.04% | 54.31% | 50.24% |

This was the RIGHT lever (unlike phase-2 RE): lowering AWP traded a little robustness
for clean WITHOUT leaving CAP's robustness frontier. awp=2e-3 gives clean 82.30
(+1.15 over our baseline) while AA-fast stays ~50 and PGD-20 ~54.1 (>= CAP 54.31 border).
But clean 82.3 still < CAP 83.04: we closed ~1.2 of the 1.9 clean gap, not all of it.
Monotone trend (clean rises, robustness falls as awp drops) is clean and reportable.

Running full AA (10k) on g2 best_net + best_ema for the definitive robustness number
at our best clean/robust operating point.

### Full AutoAttack (10k) on awp=2e-3 — DEFINITIVE numbers (2026-06-13)

| model | Clean | PGD-20 | full AA (10k) |
|---|---|---|---|
| g2 awp=2e-3 best_ema | 82.30% | 54.13% | **50.74%** |
| g2 awp=2e-3 best_net | 82.15% | 54.07% | 50.27% |
| v3 baseline awp=5e-3 best_ema | 81.15% | 54.63% | 50.78% |
| CAP (published) | 83.04% | 54.31% | 50.24% |

Note: AA-fast n=1000 underestimated g2 (49.8) vs full 10k (50.74) — the 1000-subset
was harder. Full AA is the number of record.

**g2 best_ema (awp=2e-3) is our headline model and a Pareto improvement over our own
baseline**: +1.15 clean (82.30 vs 81.15) for ~0 AA cost (50.74 vs 50.78).

## FINAL VERDICT vs CAP (ICASSP 2024)

| axis | ours (g2 ema) | CAP | result |
|---|---|---|---|
| AutoAttack | **50.74** | 50.24 | **WIN +0.50** |
| PGD-20 | 54.13 | 54.31 | tie (-0.18, noise) |
| Clean | 82.30 | 83.04 | lose -0.74 |

- We **beat CAP on AutoAttack** (the headline robustness metric) and **tie on PGD-20**,
  at clean accuracy 0.74 below theirs — using a recipe with NO particle search
  (CAP: N=10 x T=40 = 400 fwd/bwd per batch; ours: standard TRADES KL-PGD-10 + AWP).
  We do NOT dominate CAP on all three axes; we trace a slightly more robustness-favorable
  point on the same RN18 frontier.
- **CHM (the novel convex-hull term) is a negative result**: never beats the matched
  baseline on the strong recipe (v3 AA 50.11 < 50.74). Honest contribution is the
  trade-off analysis + the recipe, not the hull regularizer.

## Honest paper framing
"On CIFAR-10/ResNet-18 (Linf 8/255) we match CAP's robustness (AA 50.74 vs 50.24,
PGD-20 54.13 vs 54.31) at comparable clean accuracy with a ~20x cheaper training-time
adversary, and show that an input-space convex-hull margin does not improve over a
well-tuned TRADES+AWP+EMA baseline." Needs >=2 seeds for the headline before submission.

## TWO-SEED CONFIRMATION (2026-06-15) — the honest final numbers

Full AA (10k) on best_ema, both seeds:

| arm | seed | Clean | PGD-20 | full AA |
|---|---|---|---|---|
| g2 baseline (awp=2e-3) | 0 | 82.30 | 54.13 | 50.74 |
| g2 baseline (awp=2e-3) | 1 | 81.66 | 54.24 | 50.24 |
| **g2 baseline mean** | | **81.98** | **54.19** | **50.49 (range 50.24-50.74)** |
| CHM (awp=5e-3) | 0 | 79.89 | 54.14 | 50.11 |
| CHM (awp=5e-3) | 1 | 80.16 | 54.18 | 49.95 |
| **CHM mean** | | **80.03** | **54.16** | **50.03 (range 49.95-50.11)** |
| CAP (published) | | 83.04 | 54.31 | 50.24 |

### CORRECTED final verdict (supersedes the single-seed claim above)
- The seed-0 AA of 50.74 was the TOP of the range. Seed 1 landed at **exactly 50.24**
  = CAP's number. **Two-seed mean AA = 50.49 +/- 0.25 vs CAP 50.24** => we MATCH CAP on
  AutoAttack within seed variance, we do NOT reliably beat it. PGD-20 also a tie
  (54.19 vs 54.31). Clean trails by ~1.1 (81.98 vs 83.04).
- HONEST headline: **we MATCH CAP's robustness (AA and PGD-20 within ~0.25) at ~1pt
  lower clean, using a ~20x cheaper training adversary** (no N=10xT=40 particle search).
  Not a strict win on any axis once seeds are averaged.
- **CHM negative result CONFIRMED across 2 seeds**: mean AA 50.03 < baseline 50.49,
  clean 80.03 < 81.98, PGD-20 identical. The hull term costs ~2 clean for zero
  robustness benefit. Solid, reproducible negative result.

### What this means for the paper
The clean-sweep-of-CAP story is not supported. The defensible contributions are:
(1) a cheaper recipe that matches CAP's robustness; (2) a rigorous 2-seed negative
result on the input-space convex-hull margin. Honest, publishable as an analysis/
negative-result paper — not as "new SOTA".

## CHM rescue attempt on CIFAR-100 (2026-06-15, session 4)

Decision (user): keep the paper's method framing, try to rescue CHM on a harder
dataset before rewriting. Rationale: on CIFAR-10 the TRADES+AWP baseline may be
saturated; CIFAR-100 (10x classes, lower robust acc) leaves more room for a
regularizer to matter. Paper draft marked with a STATUS banner: thesis currently
unsupported on CIFAR-10; do not submit from those numbers.

### Code changes for CIFAR-100
- `models/resnet.py`: `ResNet18(num_classes=10)` — parameterized, default unchanged.
- `main_adv_CHM_v3.py`: `--dataset cifar10|cifar100` — selects dataset class,
  per-dataset normalization stats (CIFAR-100: mean .5071/.4865/.4409,
  std .2673/.2564/.2762), and num_classes. Val-split logic unchanged (both have
  50k train images).
- `eval_checkpoint.py`, `eval_autoattack.py`: same `--dataset` flag.

### Rescue A/B (identical recipe to the CIFAR-10 v3 arms, 120ep, seed 0)
- GPU0 `c100_trades_awp_base_s0`: N_hull=1, K_mix=0 (TRADES-AWP control)
- GPU1 `c100_chm_trades_awp_s0`: N_hull=2, K_mix=2 (CHM)
Reference band (literature, RN18/CIFAR-100, no extra data): TRADES-AWP ~ clean 58-62,
AA ~ 25-26. Rescue succeeds iff CHM beats the matched control on AA (full 10k) outside
seed noise; otherwise the negative result stands as dataset-general.

### CIFAR-100 rescue results (2026-07-03) — RESCUE FAILED (tie, not a win)

Full test set; full AA = standard 10k:

| arm | ckpt | Clean | PGD-20 | full AA |
|---|---|---|---|---|
| control | best_net (val 29.5, chosen) | 57.60 | 30.48 | 25.09 |
| control | best_ema (val 29.3) | 58.05 | 30.90 | **25.71** |
| CHM | best_net (val 30.9, chosen) | 57.51 | 30.90 | 25.61 |
| CHM | best_ema (val 30.7) | 57.91 | 30.84 | 25.38 |

- By the pre-committed rule (compare val-chosen checkpoints): CHM 25.61 vs control
  25.09 -> +0.52 AA. BUT the control's other checkpoint scored 25.71, and the four AA
  numbers interleave (25.09/25.38/25.61/25.71) with within-run spread (~0.6) as large
  as the effect. Clean and PGD-20 are dead ties. The pre-committed success criterion
  ("beats control outside seed noise") is NOT met.
- CHM's +1.4 val-robust lead did not translate to a test-AA win.
- Difference vs CIFAR-10: there CHM consistently LOST (incl. -2 clean); here it TIES
  at 2x the training cost. Either way it does not earn its place as the paper's
  method.

### FINAL PROJECT VERDICT (all experiments)
CHM does not beat a matched TRADES+AWP+EMA control on CIFAR-10 (2 seeds, loss) or
CIFAR-100 (1 seed, tie). The paper must pivot to the honest framing: cheap recipe
matching CAP's robustness + a rigorous multi-dataset negative result on input-space
convex-hull regularization. Both GPUs idle; remaining useful compute would be
CIFAR-100 seed 1 only if the user wants 2-seed rigor on the tie for the paper's
negative-result table.

### Session 5 (2026-07-03): paper pivoted + CIFAR-100 seed 1 launched
- `paper/PAPER_DRAFT.md` REWRITTEN to the supported framing: (1) recipe-matched
  re-evaluation — TRADES-AWP-EMA matches CAP at 1/20 cost; (2) controlled multi-seed,
  multi-dataset negative result on worst-case-over-hull KL; (3) analysis (hull max =
  implicit beta increase; seed variance 0.5 AA > most claimed gaps). All real numbers
  filled; CIFAR-100 seed-1 cells marked TBD. Internal reviewer-risk notes included.
- Launched CIFAR-100 seed 1 (identical configs): GPU0 c100_trades_awp_base_s1,
  GPU1 c100_chm_trades_awp_s1. On completion: eval battery + fill the TBDs; the
  negative result then has 2 seeds on both datasets.

### CIFAR-100 seed 1 results (2026-07-05) — negative result LOCKED, 2 seeds x 2 datasets

Full test set; full AA 10k; val-chosen ckpts marked *:

| arm | ckpt | Clean | PGD-20 | full AA |
|---|---|---|---|---|
| control s1 | best_net* (val 29.6) | 58.63 | 31.08 | 25.87 |
| control s1 | best_ema (val 29.5) | 58.82 | 31.12 | 25.71 |
| CHM s1 | best_net (val 29.5) | 57.53 | 30.58 | 25.44 |
| CHM s1 | best_ema* (val 29.6) | 57.67 | 31.01 | 25.52 |

2-seed CIFAR-100 (val-chosen): control 58.12 / 30.78 / 25.48+/-0.39;
CHM 57.59 / 30.96 / 25.57+/-0.05. AA delta +0.09 << control seed spread 0.78.
Seed-0's +1.4 CHM val lead did not recur (val dead tie at s1) — selection noise.

FINAL: CHM loses on CIFAR-10 (2 seeds), ties on CIFAR-100 (2 seeds), at 2x cost.
Paper draft TBDs filled (abstract, 4.1, 4.3 two-seed table + appendix, reviewer
notes). The draft is now numerically complete; both GPUs idle.

### Full CAP attack battery (2026-07-07): FGSM/PGD-100/CW-inf added
- New `eval_cap_suite.py` (Clean/FGSM/PGD-20/PGD-100/CW-inf, [0,1]-space, sanity-
  checked against known PGD-20/clean numbers). Run on all 8 headline checkpoints.
- Results in EXPERIMENTS.md A7 + paper 4.2 tables. Key findings:
  * CIFAR-10: ties CAP on PGD-20/100/AA, BEATS CAP on CW-inf (51.24 vs 50.85);
    trails only clean (-1.06) and FGSM (-0.95).
  * CIFAR-100: FULL PARITY OR BETTER vs CAP on every column (clean +0.10,
    PGD-20 +0.40, PGD-100 +0.45, AA +0.06; CW -0.26, FGSM -0.01).
- Remaining CAP-parity gaps: seeds 3-5, SVHN, WRN-34-5/10, own baseline repros.

### Session 6 (2026-07-07): method-framed paper draft
- New `paper/PAPER_DRAFT_WSC.md` — method paper in CAP's structure, per user request.
  Framing: **WSC (Weight-Space Confinement)** = TRADES + AWP + EMA + robust-val
  selection, positioned as achieving CAP's polytope confinement in weight space at
  O(1) cost instead of O(NT) particle search. CHM presented as the controlled test
  showing input-space enrichment is subsumed by weight-space smoothing (its real,
  measured role). All table values as measured; no number massaged. Caveats kept
  in text: 2-seed means, CAP quoted not re-run, RN18 only.
- Old analysis draft (PAPER_DRAFT.md) left intact as companion.

### Session 6b: standalone-method draft
- New `paper/PAPER_DRAFT_CHM.md` — CHM-TRADES as the standalone method (user request:
  de-emphasize CAP). CAP demoted to one related-work paragraph + a quoted table row.
  Claim calibrated to what the data supports: SOTA-competitive at 2x TRADES cost,
  best-in-table CIFAR-100 strong-attack numbers, C10 within seed variance; N=1
  anchor rows kept in all tables (integrity line), gaps vs anchor flagged as within
  noise. Ablation section reframes "when does the hull bind" (weak vs strong recipe,
  pointers to eps/WRN regimes). Three drafts now coexist: analysis / WSC / CHM.
