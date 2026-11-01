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
