# Session Strategy Notes

## Purpose

This file fixes the main ideas we already tested, the outcomes of those checks, and the exact strategy of the latest session that gave the current best result.

## Current Best Result

- Local `dev_cer`: `0.0738`
- Local `indomain_cer`: `0.0597`
- Local `ood_cer`: `0.0789`
- Local `harmonic_cer`: `0.0680`
- Kaggle public score: `6.835`
- Current public leaderboard place: `5 / 8`

## Working Core Strategy

Use a compact word-level CTC ASR model for Russian spoken numbers:

- input: waveform resampled to `16 kHz`
- frontend: self-contained `log-mel`
- encoder: `Conv + BiGRU`
- decoder: greedy CTC
- post-processing: deterministic word-to-number reconstruction

Why this remains the base approach:

- the output grammar is tiny and structured
- word-level decoding is easier to regularize than direct digit prediction
- deterministic reconstruction keeps inference simple for Kaggle
- the model stays under the `<5M` parameter limit

## Ideas Already Checked

### 1. Local partial test submission

What we tried:

- local submit from incomplete downloaded `test`

Outcome:

- not useful for model evaluation
- produced bad score because most test rows had fallback values

Decision:

- do not use partial local `test` for evaluation
- full evaluation must run on Kaggle-hosted competition `test`

### 2. Baseline word-level CTC without extra decoding tricks

What we tried:

- compact `Conv + BiGRU + CTC`
- greedy decode
- best-effort numeric reconstruction

Outcome:

- first correct full-test Kaggle submission: `27.122`
- baseline was valid and transferred to leaderboard

Decision:

- keep this as the base system

### 3. Continue training from the first useful checkpoint

What we tried:

- continue training from existing checkpoint instead of redesigning architecture

Outcome:

- local metrics improved strongly
- Kaggle public score improved from `27.122` to `12.061`

Decision:

- longer training was worth more than early architectural changes

### 4. Error analysis by speaker

What we checked:

- per-speaker CER on `dev`
- focus on unseen speakers, especially `spk_I` and `spk_K`

Outcome:

- hardest errors were dominated by:
  - `thousands_or_higher`
  - `length_mismatch`
- this indicated robustness problems in tempo/speaker variation, not only local digit mistakes

Decision:

- next gains should come from regularization and augmentation before beam search or LM ideas

### 5. Mild speed perturbation + SpecAugment

What we tried:

- waveform speed perturbation with small rate range
- time masking and frequency masking on log-mel features

Outcome:

- local `dev_cer`: `0.1182 -> 0.0738`
- `spk_I`: `0.1479 -> 0.0713`
- `spk_K`: `0.1789 -> 0.1027`
- Kaggle public score: `12.061 -> 6.835`

Decision:

- this is the best strategy tested so far
- keep it as the current baseline branch

## Last Successful Session Strategy

This is the exact logic of the session that produced the best result so far.

### Step 1. Diagnose the real bottleneck

We did not start from decoding complexity.
We first inspected `dev` by speaker and found that the biggest remaining damage was on OOD speakers, especially `spk_I` and `spk_K`.

Main observation:

- many errors were not just single-digit substitutions
- there were many failures in higher-order digits and output length

Practical interpretation:

- the model needed better robustness to tempo, speaking style, and acoustic variation

### Step 2. Add only low-risk regularization

We kept the architecture and inference pipeline unchanged.
Instead, we added:

- mild waveform speed perturbation
- time masking on spectrogram frames
- frequency masking on mel bins

Why this was chosen:

- it directly targets speaker and tempo variation
- it does not complicate Kaggle inference
- it is cheap compared with redesigning the decoder

### Step 3. Train from the best existing checkpoint

We resumed from the strongest checkpoint instead of retraining from scratch.

Why:

- faster iteration on CPU
- preserves already learned number grammar
- lets us measure the effect of augmentation more cleanly

### Step 4. Validate locally before Kaggle

We checked:

- overall `dev_cer`
- `indomain_cer`
- `ood_cer`
- `harmonic_cer`
- per-speaker CER for hardest speakers

This was important because the competition metric is sensitive to out-of-domain speakers, so total CER alone is not enough.

### Step 5. Run full Kaggle evaluation only after local confirmation

After local improvement, we:

- uploaded the new checkpoint to a Kaggle dataset
- reran the private inference kernel on the full competition `test`
- submitted the produced `submission.csv`

This prevented us from guessing based only on local metrics.

## Why This Strategy Worked Best So Far

- it attacked the real error mode: OOD speaker generalization
- it improved robustness without adding decoding fragility
- it preserved the simple and reproducible Kaggle inference path
- it transferred to public leaderboard instead of improving only on local `dev`

## Current Baseline To Reuse

If we want to restart from the current best known branch, the default recipe is:

1. Word-level CTC model with deterministic number reconstruction.
2. Train from the best checkpoint, not from scratch, unless we explicitly test that variable.
3. Keep gain/noise augmentation.
4. Keep mild speed perturbation.
5. Keep log-mel time/frequency masking.
6. Judge new ideas primarily by `ood_cer`, `harmonic_cer`, and Kaggle full-test score.

## Next Narrow Ideas

The next candidates should stay close to the current successful branch:

- tune masking intensity, not replace the whole augmentation scheme
- tune speed perturbation range carefully
- inspect remaining `spk_K` failures before adding decoding complexity
- only then test constrained decoding or small grammar-aware repair

## Ideas To Avoid For Now

- evaluating quality from partial local `test`
- jumping to heavy decoding changes before checking OOD generalization
- changing architecture first when current bottleneck is still speaker robustness
