# Sprint notes: beating 50.49 AA (started 2026-07-07)

Plan of record = IDEAS_BEAT_SOTA.md "attack plan". Every action logged here with
date, what/why, and outcome. Baseline to beat: RN18 C10 eps=8/255 full AA 50.49
(2-seed); best single seed 50.78 (v3_trades_awp_g2_s0_best_ema).

## Step 1 — MeanSparse post-processing (free, no training) [IN PROGRESS]
Ref: arXiv:2406.05927 (post-training mean-centered feature sparsification, used by
2024-25 RobustBench top entries).
- Operator: y = mu + (x - mu) * 1[|x - mu| > alpha * sigma], per channel, after each
  ReLU; mu/sigma calibrated on training data with the checkpoint frozen.
- Implementation: new `models/resnet_ms.py` (ResNet18MS = RN18 + MeanSparse modules
  after every ReLU; loads plain RN18 checkpoints with strict=False). New
  `eval_meansparse.py`: (1) calibrate stats on train batches, (2) select alpha by
  PGD-10 robust acc on our held-out 1k val split (NOT test), (3) evaluate test
  PGD-20 + AA-fast at chosen alpha; full AA only if it looks like a gain.
- Masking caveat to check: the indicator has zero gradient in the "off" region ->
  report Square (black-box) inside AA specifically; a gain that appears only in
  white-box attacks is masking, not robustness.
- Target checkpoint: v3_trades_awp_g2_s0_best_ema (AA 50.78).

## Step 2 — ReBAT + Swish recipe floor [PENDING]
Fetch exact ReBAT tricks from github.com/PKU-ML/ReBAT, implement v4 script, launch
A/B overnight (both GPUs).

## Step 3 — AdaAD distillation [PENDING]
Need a robust WRN teacher checkpoint from RobustBench zoo (download size ~200MB).

## Step 4 — Hetero-hull CHM on best recipe [PENDING]
Vertices from different attack losses (CE/CW/KL). Instrument vertex-diversity metric.
