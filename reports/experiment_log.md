# Experiment log

## 2026-11-01 — Day 1 bootstrap

### Решения

- Стек сдачи: Colab T4 + TRL/PEFT/bitsandbytes (не MLX dual-path).
- Источник: оригинальный Gardening SE dump; SHA1 совпал с аудитом.
- Spot-review: 25 примеров; denylist 4 off-topic keep; 1 ложный drop зафиксирован.
- Judge: `openai/gpt-4o-mini` через OpenRouter; smoke OK (total=8 на эталонном ответе, cost≈$0.00012).

### Данные

| Этап | N |
|---|---:|
| Вопросы в dump | 17860 |
| Expanded tags + positive best answer | 2802 |
| Technical (no img, not closed) | 839 (audit 838) |
| After sufficiency heuristics | 825 |
| После denylist (ожид.) | train≈741–745 |

Splits seed=42: plant_dev/test = 40/40 groups; train remainder.

### Проверки

- `pytest`: 11 passed (checker pass/fail, split leakage, score, html/tags).
- `datasets.load_dataset` на plant JSONL — ок (при `HF_HOME` в workspace).
- Baseline generation: dry-run промптов сохранён; полный прогон базы — в Colab / локально с transformers.

### Открыто

- Полный train S1 на T4.
- Массовый judge на plant_dev subsample.
- Baseline/SFT генерации и таблицы метрик.
- Pin revision модели после первого успешного download.

## 2026-11-01 — Colab API fix (transformers v5)

### Симптомы на T4

1. `TrainingArguments ... unexpected keyword argument 'warmup_ratio'`
2. `SFTTrainer ... unexpected keyword argument 'tokenizer'`

Причина: `%pip install -U` подтянул transformers v5 / новый TRL.

### Исправление

- `configs/train.yaml`: `warmup_ratio` (v4); код мапит на float `warmup_steps` для v5
- `src/train/sft_qlora.py`: `SFTConfig` + `processing_class` + `peft_config`, native `messages`
- `requirements-colab.txt` pinned: transformers==4.57.1, trl==0.24.0, peft==0.17.1
- Локальный smoke на Python 3.12 (`.venv-colab`): `scripts/smoke_colab_stack.py` → SMOKE_OK
- Не использовать peft 0.17 с transformers 5 (HybridCache ImportError)
- Не использовать `--force-reinstall -r requirements-colab.txt` в Colab: ломает numpy/torch (`_slice` ImportError). Ставить через `scripts/colab_install.sh`; при поломке — Disconnect and delete runtime.
- Colab CUDA 13: `bitsandbytes==0.45.5` ставит CPU wheel → `cquantize_blockwise_fp16_nf4` AttributeError. Нужен `bitsandbytes>=0.49.2` с `libbitsandbytes_cuda130.so`.
- `assistant_only_loss=True` ломает Qwen2.5 (нет `{% generation %}` в chat template). Для pilot/S1: `assistant_only_loss: false` (loss по всей последовательности).

## 2026-11-01..03 — Pilot, S1, eval

- Pilot 20 steps: loss 2.95→2.29, save/reload + генерация OK (3 пункта).
- S1: 1 epoch, 93 steps, 870 s T4; loss 3.04→2.08; eval_loss plant_dev 2.07 (step 50). Адаптер + checkpoint-50/93 на Drive.
- Первый eval упал FileNotFound: не было шага генерации → `scripts/run_eval_pipeline.py` (resumable, пишет на Drive).
- Исправлено до прогона: checker принимал 5 шагов; judge кэшировал ошибки.
- Eval 2026-11-03 (~54 мин): dev judge base 6.54 / S1 3.15 / step50 2.97; format 0.92 / 0.18 / 0.23; general 0.825 / 0.863 / 0.863.
- Правило: оба SFT-чекпоинта не прошли (plant хуже) → selected = base.
- Test: judge 6.64 → 3.67, format 0.97 → 0.13, general 0.819 → 0.863.
- Причина (по данным train): нумерованные списки у 3% ответов, «I think / I'm no expert» у 20%.
- S2 не запускался: регрессии general нет, проблема в plant-данных.

## 2026-11-04 — Разбор результатов

- Гипотеза «SFT не останавливается» снята: медиана токенов plant test base 376 / SFT 109; медленные батчи — ожидание самого длинного ответа в батче.
- Длина не объясняет разрыв: corr(len, judge) внутри SFT −0.25 dev / 0.09 test; попарно SFT 0 побед из 39 (dev и test).
- Судья по самим целям SFT (ответы SE, dev, без ориентира): 4.79 vs base 7.00 vs S1 3.36 (`scripts/judge_references.py`). Данные хуже базы → причина регрессии.
- Качественные ошибки SFT: уверенные фактические ошибки, выдуманный «личный опыт», зацикливание, игнор формата (30/34 провалов — ни одного пункта).
- General +4.4 п.п. — в основном форма ответа (H₂O, «только число») и units, не рассуждения.
- План S2: гейт качества целей → сузить тему → переписанные/отфильтрованные цели + assistant-only loss + format-примеры.
