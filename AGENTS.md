# AGENTS.md — Houseplant SFT (RK1, Colab-first)

## 0. Статус

- Версия: **0.2 — Colab-first план РК1**.
- Дата: **2026-11-01**.
- Исследование данных: `DATASET_RESEARCH.md`, `DATASET_AUDIT.json` (источник правды по счётчикам).
- Обучение и финальные метрики ещё не прогнаны; сдаваемый стек — **Colab T4**.

## 1. Задача РК1

**Англоязычный текстовый QA-помощник по уходу за комнатными/горшечными декоративными растениями**, с жёстким контролем сохранения общих способностей.

Обязательный комплект сдачи:

1. Тема и задача зафиксированы.
2. 1–2 baseline-ответа базы сохранены.
3. Одна автопроверка + pass/fail примеры.
4. LLM-as-judge (OpenRouter).
5. SFT JSONL.
6. Дообучение + LoRA safetensors.
7. Сравнение base vs SFT: plant **и** general.
8. Репозиторий, отчёт 5–10 стр. (RU), код, данные, графики, метрики.

Код сдачи работает на **T4 в Google Colab** (`transformers` + `TRL` + `PEFT` + `bitsandbytes`). Mac M5 Pro 24GB — подготовка данных/скриптов и локальный smoke, не основной train.

## 2. Зафиксированные решения

| Решение | Выбор |
|---|---|
| База | `Qwen/Qwen2.5-3B-Instruct` |
| Данные | Оригинальный dump Gardening SE |
| Старт-фильтр | expanded tags → best Score>0 → без `<img>` → не closed (~838), затем эвристики text-sufficiency |
| Spot-review | 25 разнотипных keep/drop (оценка yield, не полный ручной корпус) |
| Splits | group split по `question_id`; plant_dev/test по 30–50; train ~400–700 после автофильтра |
| General | `general_dev` 80 / `general_test` 160; не входят в SFT |
| Судья | OpenRouter, фиксированная модель, рубрика 4×(0–2) = 0–8 |
| Язык | данные EN; отчёты RU |

**Non-goals РК1:** RAG, веб, мультимодальность, перевод корпуса, dual-path MLX↔PEFT, EcoSage/синтетика в первом прогоне, обучение отказам «только про растения».

## 3. Данные

Источник: [gardening.stackexchange.com.7z](https://archive.org/download/stackexchange/gardening.stackexchange.com.7z)  
SHA1 архива: `718db7040d0b608b1c969d4630d8bb95db87eec5`.

Expanded tags: `houseplants`, `indoors`, `container-gardening`, `containers`, `pots`, `succulents`, `cactus`, `orchids`, `ficus`, `african-violets`, `aloe`, `dracaena`, `peace-lily`, `monstera`, `philodendron`, `jade`, `spider-plant`.

Контракт JSONL (одна запись = один пример):

```json
{
  "id": "gardening-q{qid}-a{aid}",
  "group_id": "gardening-q{qid}",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "source": "gardening.stackexchange.com",
    "question_id": 0,
    "answer_id": 0,
    "source_urls": [],
    "licenses": [],
    "language": "en",
    "tags": [],
    "answer_origin": "human",
    "text_sufficient": true,
    "review_status": "auto_accepted"
  }
}
```

Метаданные можно вынести в sidecar; runtime для SFT читает `messages`. Загрузка через `datasets.load_dataset("json", ...)`.

## 4. Оценивание и выбор чекпоинта

**Автопроверка (главная для задания):** plant-промпт просит «3 numbered steps» → ровно 3 непустых пункта `1./2./3.` без дублирования одного совета.

**General suite:** 6 категорий коротких задач (arith, units, extract, logic, JSON/schema, stable facts) + ARC-Easy subset (dev←validation, test←test). Метрики: overall + per-category accuracy; correct→wrong / wrong→correct; refusals вне темы.

**Правило выбора чекпоинта (жёстко):**

1. `plant_dev` не хуже базы (или лучше).
2. `general_dev` не падает **больше чем на 5 п.п.** относительно базы.
3. При конфликте — более мягкий чекпоинт / короче train; если ничего не проходит — оставить base и честно описать в отчёте.

**Judge:** факты / контекст / практичность / калиброванная неопределённость; critical-error flag отдельно. Ключ только в `OPENROUTER_API_KEY`. Без silent provider fallback в сравниваемой серии.

База и адаптер сравниваются с одинаковыми: model id/revision, tokenizer, chat template, system prompt, decoding, quantization backend.

## 5. Обучение (Colab T4)

- B0: база без адаптера.
- S1: один основной QLoRA SFT на `plant_train`.
- S2: только при регрессии general — меньше steps/LR **или** ~10–20% `general_replay_train` (не из dev/test).

Ориентиры после pilot: seq 512–1024, LoRA rank 8–16, LR ~1e-4, 1–2 эпохи. Pilot 10–20 steps → save/reload → 1 генерация, затем S1.

## 6. Структура репозитория

```text
AGENTS.md
README.md
DATASET_RESEARCH.md
DATASET_AUDIT.json
configs/
src/data/
src/eval/
src/train/
scripts/
notebooks/rk1_colab.ipynb
tests/
data/raw/          # gitignored
data/processed/
reports/
```

## 7. Роли

- **Coordinator:** этот файл, README, отчёт, запрет тюнинга по test.
- **Data:** `src/data/`, manifests, plant JSONL.
- **Eval:** `src/eval/`, general suite, judge, aggregation.
- **Train:** `src/train/`, Colab notebook, save/reload.

Один `run_id` на эксперимент. Не заполнять неизвестные поля догадками в отчёте.
