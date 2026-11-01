# Отчёт РК1 (черновик): SFT помощника по комнатным растениям

**Статус:** черновик по ходу Day 1; обучение и финальные метрики будут дописаны после Colab S1.

## 1. Постановка

Тема: англоязычный текстовый QA по уходу за комнатными и горшечными декоративными растениями.  
Гипотеза: небольшое тематическое QLoRA-SFT улучшает полезность ответов о растениях при падении general accuracy не больше чем на 5 п.п. на `general_dev`.  
База: `Qwen/Qwen2.5-3B-Instruct`. Среда сдачи: Google Colab T4.

## 2. Данные

Источник — dump Gardening Stack Exchange (archive.org), SHA1 `718db704…`.  
Фильтр: expanded tags → best answer Score>0 → без `<img>` → не closed → эвристики text-sufficiency.  
Получено **825** auto-keep (technical ≈839 ≈ audit 838). Spot-review 25: см. `reports/spot_review_25.md`; denylist 4 off-topic ID.  
Разбиение по `question_id`, seed 42: train / dev40 / test40.

## 3. Baseline (шаг задания)

Промпты зафиксированы в `configs/eval.yaml` → `reports/metrics/baseline_plant_samples.jsonl`.  
Полные ответы базы добавляются после прогона на T4/MPS (`scripts/generate_baseline.py`).

## 4. Автопроверка

Контракт: при явной просьбе о трёх нумерованных шагах ответ содержит ровно `1./2./3.` непустых различных пунктов (`src/eval/checkers.py`).  
Обоснование: уход должен быть actionable и не зацикливаться на одном совете.  
Pass/fail: `tests/test_checkers.py` (11 тестов в suite).

## 5. LLM-as-judge

OpenRouter, модель `openai/gpt-4o-mini`, рубрика 4×(0–2), `critical_error` отдельно.  
Smoke 2026-11-01: статус ok, total=8 на калибровочном ответе. Массовый прогон — после генераций.

## 6. Обучение

План: pilot 20 steps → S1 1 epoch QLoRA r=16.  
S2 только при регрессии general.  
Артефакт: LoRA safetensors + config (вне git).

## 7. Сравнение base vs SFT

Будут таблицы: plant (judge subsample + format checker) и general (accuracy overall/per-category, transitions, refusals).  
Правило выбора чекпоинта — в `AGENTS.md`.

## 8. Выводы

*(заполнить после S1)*  
Ограничения: остаточный off-topic в SE-тегах; отсутствие фото отсекает много реальных houseplant-тредов; голоса ≠ экспертная истина.
