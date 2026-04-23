# ASR Spoken Numbers Plan

## Goal

Build a compact Russian ASR system for spoken numbers in the range `0..999999` for the Kaggle competition `asr-2026-spoken-numbers-recognition-challenge`, with a strong emphasis on robustness to unseen speakers and noisy audio.

## Constraints

- Train only from scratch on the provided data.
- Input audio must be resampled to `16 kHz`.
- Model size must stay below `5M` parameters.
- `dev` is validation only and must not be merged into training.
- Primary competition signal is CER, with special importance of out-of-domain speakers.

## Main Strategy

Use a small CTC-based ASR model that predicts **spoken number words**, not digits directly.

Why this is the default baseline:

- The output grammar is extremely constrained.
- The set of spoken tokens is small and interpretable.
- We can convert predicted words back to digits with deterministic post-processing.
- This gives us a direct path to error analysis by token pattern, number structure, and speaker.

## Working Baseline

### Representation

- Label normalization:
  - `139473 -> "сто тридцать девять тысяч четыреста семьдесят три"`
  - `1005 -> "одна тысяча пять"`
- Output vocabulary:
  - units, teens, tens, hundreds
  - `тысяча / тысячи / тысяч`
- Post-processing:
  - greedy CTC decode
  - deterministic parser from words back to integer
  - best-effort recovery for mildly invalid token sequences

### Model

- Log-mel frontend at `16 kHz`
- Compact encoder:
  - convolutional temporal subsampling
  - bidirectional GRU stack
  - linear CTC head
- Hard cap check for parameter count `< 5M`

### Validation

- Overall CER on digit strings
- CER by `spk_id`
- Structured error review:
  - thousands handling
  - teen confusion
  - omitted suffixes / endings
  - failures on noisy samples

## Iterations

### Iteration 0. Infrastructure

Status: `completed`

- Create project structure.
- Implement text normalization and inverse parsing.
- Implement audio loading and 16 kHz resampling.
- Implement baseline dataset, model, training loop, inference script.
- Add unit tests for number text logic.

Exit criteria:

- Code runs end-to-end on local CSV/audio layout.
- We can train a baseline and produce a Kaggle-ready prediction CSV.

Current state:

- Project skeleton is implemented.
- Number normalization and inverse parsing are implemented and covered by unit tests.
- Baseline training, inference, submission, and data inspection scripts are implemented.
- Local `train/dev` data is present and validated.
- A first short baseline run was executed on real data with bounded train/dev batches to get initial metrics.
- Kaggle full-test inference is now automated through a private weights dataset plus a private competition kernel.
- Remaining limitation: the local environment is CPU-only (`cuda=False`, `mps_available=False`), so large training iterations are slower than ideal.

### Iteration 1. Strong Baseline

Status: `completed`

- Train baseline without external LM.
- Check dev CER and per-speaker spread.
- Inspect top failure modes manually.
- Verify stability on mixed `wav/mp3`, varying sample rates, and noisy clips.

Exit criteria:

- Stable training.
- Reproducible checkpoint and inference pipeline.
- First submission to leaderboard.

Current state:

- First correct full-test Kaggle submission scored `27.122` on `2026-04-12`.
- After continuing training for one more full epoch from `last.pt`, local metrics improved to:
  - `dev_cer=0.1182`
  - `indomain_cer=0.0948`
  - `ood_cer=0.1266`
  - `harmonic_cer=0.1084`
- The second full-test Kaggle submission scored `12.061` on `2026-04-12`.
- Current public leaderboard position is `5 / 8`.

### Iteration 2. Generalization To Unseen Speakers

Status: `in progress`

- Increase regularization only where justified by dev speaker spread.
- Add waveform / feature augmentations:
  - gain
  - noise
  - time masking
  - frequency masking
  - optional speed perturbation if the toolchain permits it cleanly
- Review model width/depth against overfitting signs.

Exit criteria:

- Reduced worst-speaker CER.
- Smaller gap between easiest and hardest speakers.

Current state:

- Added mild speed perturbation and log-mel time/frequency masking to the training pipeline.
- Reusable dev error analysis is now scriptable, which makes speaker-focused iteration faster.
- After one augmented epoch from the previous best checkpoint, local metrics improved to:
  - `dev_cer=0.0738`
  - `indomain_cer=0.0597`
  - `ood_cer=0.0789`
  - `harmonic_cer=0.0680`
- Hardest OOD speakers improved substantially:
  - `spk_I: 0.1479 -> 0.0713`
  - `spk_K: 0.1789 -> 0.1027`
- The corresponding Kaggle full-test submission scored `6.835` on `2026-04-12`.
- Current public leaderboard position remains `5 / 8`, but the gap to `4` place is now only `1.181`.

### Iteration 3. Decoding Improvements

Status: `pending`

- Add constrained decoding over valid number token grammar.
- Try beam search with grammar-aware pruning.
- Optionally add LM rescoring if it improves dev and stays operationally simple.

Exit criteria:

- Clear dev gain over greedy decoding.
- No brittle inference complexity for Kaggle submission.

### Iteration 4. Targeted Error Repair

Status: `pending`

- Analyze confusion buckets:
  - `один/одна`
  - `два/две`
  - teens vs tens + units
  - dropped thousands marker
  - trailing low-order digits after `тысяча`
- Add targeted decoding heuristics or grammar repairs only if they reduce dev CER consistently.

Exit criteria:

- Fewer structurally invalid predictions.
- Measurable reduction in numeric reconstruction errors.

### Iteration 5. Submission Hardening

Status: `pending`

- Freeze best checkpoint and exact config.
- Make inference script deterministic.
- Validate submission schema and ordering.
- Prepare experiment log and report artifacts.

Exit criteria:

- Reproducible final submission package.
- Clean experiment history and documented decisions.

## Risks

- `torchaudio` is broken in the current local Python environment, so the project should not depend on it for the baseline.
- `soundfile` support for `mp3` depends on the local backend; if decoding fails, we will need either a different runtime image or a controlled conversion step.
- Kaggle metric emphasizes out-of-domain speakers, so a stronger train CER is not useful unless dev speaker spread improves.

## Decision Log

- `2026-04-11`: baseline will use a **word-level number grammar + CTC** pipeline instead of direct digit decoding.
- `2026-04-11`: frontend will avoid `torchaudio` and use a self-contained STFT/log-mel implementation on top of `torch` + `scipy` + `soundfile`.
- `2026-04-12`: real local data contains labels below `1000`, so the effective supported range must be `0..999999`, not `1000..999999`.
- `2026-04-12`: current machine does not provide usable `cuda` or `mps`, so fast iteration requires bounded runs and careful experiment sizing.
- `2026-04-12`: the correct submission workflow is Kaggle-side inference on the hosted competition `test`, not partial local test downloads.
- `2026-04-12`: continuing baseline training from the first useful checkpoint transferred well to public LB: `27.122 -> 12.061`.
- `2026-04-12`: the dominant OOD failure modes after the stronger baseline are `thousands_or_higher` and `length_mismatch`, so the next regularization step should target tempo and feature robustness before decoding complexity.
- `2026-04-12`: mild speed perturbation plus SpecAugment produced the next strong local gain without changing the inference architecture, and transferred to public LB: `12.061 -> 6.835`.

## Immediate Next Actions

1. Freeze the augmented checkpoint as the new baseline and analyze what still separates us from `5.654`.
2. Focus the next iteration on the remaining OOD speaker misses, especially `spk_K`, before attempting decoding complexity.
3. Try one narrow decoding improvement or stronger-but-still-safe augmentation tuning, then validate on full Kaggle `test`.
4. Keep architecture changes secondary until speaker-generalization gains from regularization are exhausted.
