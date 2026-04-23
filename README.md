# Русский ASR произнесённых чисел — Kaggle ASR-2026

Код и веса к решению соревнования
[asr-2026-spoken-numbers-recognition-challenge](https://www.kaggle.com/competitions/asr-2026-spoken-numbers-recognition-challenge).

Команда: **Mamarin & Solomenchuk** — Георгий Мамарин, Оксана Соломенчук.

Финальный результат: **CER 0.946** на публичной части лидерборда, **место #3**.

Полный отчёт по стратегии, экспериментам и результатам — в [STRATEGY.md](STRATEGY.md).

## Что здесь

- Word-level CTC ASR для чисел 0–999 999 (~3.5 M параметров, лимит 5 M).
- `Conv + BiGRU + CTC` + грамматический лучевой декодер.
- Аугментации на волну: mp3, полоса, сдвиг тона, изменение темпа, реверберация.
- SWA + TTA + ансамбль двух сидов на инференсе.

Полный разбор траектории от 6.835 до 0.946 и описание всех решений — в [STRATEGY.md](STRATEGY.md).

## Как запустить

### 1. Окружение

```bash
python3 -m pip install -r requirements.txt
```

Тестировалось на Python 3.12, PyTorch 2.11 (CPU).

### 2. Данные

Из-за размера данные соревнования в репозитории не лежат. Скачать с Kaggle:

```bash
kaggle competitions download -c asr-2026-spoken-numbers-recognition-challenge -p data/
unzip data/*.zip -d data/
```

Ожидаемая структура:

```
data/
  train/train.csv
  train/train/*.wav
  dev/dev.csv
  dev/dev/*.wav  data/dev/dev/*.mp3
```

### 3. Обучение

Финальный рецепт (одна модель с нуля + реверберация):

```bash
python3 scripts/train.py --config configs/scratch_reverb_seed43.yaml
```

Займёт ~9 часов на CPU (22 эпохи). Чекпоинты сохраняются в `outputs/scratch_reverb_seed43/`.

Полная воспроизводимость финального ансамбля — две тренировки:

```bash
# Сид 42: сначала с нуля 20 эпох, потом дообучение с ревербом 6 эпох
python3 scripts/train.py --config configs/scratch_full_aug.yaml
python3 scripts/train.py --config configs/reverb_cont.yaml \
    --resume-checkpoint outputs/scratch_full_aug/swa_16_18_19_20.pt \
    --fresh-optimizer --reset-best-cer

# Сид 43: с нуля 22 эпохи с ревербом с первой эпохи
python3 scripts/train.py --config configs/scratch_reverb_seed43.yaml

# Сборка SWA сида 42 (эпохи 23, 25, 26 дообучения с ревербом)
python3 scripts/average_checkpoints.py \
    --checkpoints outputs/reverb_cont/epoch{23,25,26}.pt \
    --output outputs/reverb_cont/swa_23_25_26.pt
```

### 4. Инференс

Одной моделью с TTA и грамматическим лучевым декодером:

```bash
python3 scripts/eval_dev.py \
    --config configs/reverb_cont.yaml \
    --checkpoint outputs/reverb_cont/swa_23_25_26.pt \
    --decoder beam --tta --beam-size 16
```

Ансамбль двух моделей (финальная конфигурация):

```bash
python3 scripts/eval_ensemble.py \
    --config configs/reverb_cont.yaml \
    --checkpoints outputs/reverb_cont/swa_23_25_26.pt \
                  outputs/scratch_reverb_seed43/epoch20.pt \
    --tta --beam-size 16
```

## Готовые веса

Предобученные чекпоинты раздаются через
[GitHub Releases](https://github.com/dmagog/aith_speech_recognirion_dz3/releases):

- `best.pt` — сид 42, SWA(эпохи 23, 25, 26) после реверб-дообучения. Валидационный CER 0.0152 с TTA.
- `best_seed43.pt` — сид 43, эпоха 20 обучения с нуля с ревербом. Валидационный CER 0.0128.

Ансамбль двух чекпоинтов + TTA + лучевой декодер даёт валидационный CER **0.0115**.

## Публичный Kaggle-ноутбук

Публичный ноутбук с инференсом ансамбля, импортирующий веса из GitHub Release:

- [notebooks/kaggle_inference.ipynb](notebooks/kaggle_inference.ipynb) — локальная версия.
- [kaggle_assets/full_test_kernel/run_inference.py](kaggle_assets/full_test_kernel/run_inference.py) — эквивалентный скрипт.

Опубликован на Kaggle: <https://www.kaggle.com/code/dmagog/asr-numbers-public-inference>.

### Как запустить в Kaggle

1. Откройте ноутбук по ссылке выше и нажмите **Copy & Edit** (либо через Kaggle → Code → New Notebook → Import → выбрать файл [notebooks/kaggle_inference.ipynb](notebooks/kaggle_inference.ipynb)).
2. Settings → **Internet: On** (требуется верификация телефона на Kaggle).
3. Add data → **Competition**: `asr-2026-spoken-numbers-recognition-challenge`.
4. Run All.

Если интернет недоступен, ноутбук автоматически ищет веса в `/kaggle/input/**/best.pt` и `/kaggle/input/**/best_seed43.pt`. В этом случае дополнительно нужно подключить Kaggle Dataset с весами (например, скачать веса из GitHub Release и загрузить их в приватный Kaggle Dataset).

## Структура проекта

```
src/asr_numbers/
  audio.py            — загрузка аудио, ресэмплинг через scipy
  augment.py          — mp3, полоса, тон, реверберация (все на numpy)
  dataset.py          — NumbersDataset + WeightedRandomSampler
  decoder.py          — жадный + CTC-prefix-beam с грамматикой
  features.py         — самодостаточный log-mel + SpecAugment с относительными масками
  metrics.py          — CER, CER по говорящим и по доменам
  model.py            — ConvGRUCTCModel, ~3.5 M параметров
  text.py             — число ↔ слова (мужской / женский / нейтральный)
  tta.py              — TTA через усреднение log_softmax
  vocab.py            — словарь (42 слова + blank)

scripts/
  train.py                   — цикл обучения + косинусный LR + resume
  eval_dev.py                — оценка одной модели (greedy / beam / TTA)
  eval_ensemble.py           — оценка ансамбля + TTA + beam
  analyze_dev_errors.py      — разбор ошибок по говорящим
  average_checkpoints.py     — SWA
  sweep_inference.py         — перебор beam × TTA
  make_report_figures.py     — графики для отчёта
  collect_speaker_cers.py    — сбор per-speaker CER для milestones
  submit_from_kernel.sh      — скачать output Kaggle-кернела и сабмитнуть

configs/                     — YAML-конфиги всех прогонов
kaggle_assets/
  full_test_kernel/          — Kaggle-кернел инференса
notebooks/
  kaggle_inference.ipynb     — публичный Jupyter-ноутбук
outputs/report_figures/      — графики для отчёта
tests/                       — unit-тесты text.py
STRATEGY.md                  — подробный отчёт о стратегии и результатах
```

## Лицензия

MIT, см. [LICENSE](LICENSE).
