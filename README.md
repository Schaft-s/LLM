# Houseplant SFT (RK1)

Англоязычный QA по комнатным/горшечным растениям на `Qwen/Qwen2.5-3B-Instruct` с QLoRA на **Colab T4**. Локально готовятся данные и оценки; обучение — в `notebooks/rk1_colab.ipynb`.

## Быстрый старт (локально)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # затем OPENROUTER_API_KEY=...

# данные уже собраны в data/processed/; пересборка:
# python scripts/prepare_data.py
# python scripts/apply_spot_denylist.py
# python scripts/build_general_suite.py

pytest -q
python scripts/generate_baseline.py --dry-run
python scripts/run_judge.py --input data/processed/plant_dev.jsonl --limit 1 --dry-run
```

SFT v2 (полный дамп, все ответы, отбор судьёй; разбор — `reports/sft_v2_data.md`):

```bash
.venv-colab/bin/python scripts/build_sft_pool.py   # нужен токенизатор Qwen
python scripts/judge_sft_pool.py --calibrate         # судья отбора на dev
python scripts/judge_sft_pool.py                     # ~$3.3, кэш в data/judge_cache_sft/
python scripts/select_sft_data.py                    # → data/sft_v2/sft_train.jsonl
```

Сырой dump: `data/raw/` (gitignored). SHA1 архива: `718db7040d0b608b1c969d4630d8bb95db87eec5`.

## Colab T4

Один ноутбук `notebooks/rk1_colab.ipynb` (Runtime = GPU T4, Secret `OPENROUTER_API_KEY`, Run all):

1. clone/`git pull`, затем `bash scripts/colab_install.sh` (пины 4.57/trl/peft; без `--force-reinstall`).
2. Pilot: `python scripts/train_qlora.py --pilot` (save/reload + 1 generation).
3. S2: `python scripts/train_qlora.py` на `data/sft_v2/sft_train.jsonl` → копия адаптера в `MyDrive/rk1_houseplant/s2_sft_v2`; если адаптер там уже есть, обучение пропускается.
4. `scripts/run_eval_pipeline.py`: base vs adapter на `plant_*`, `garden_*`, `general_*` одинаковыми decoding settings, результаты в `MyDrive/rk1_houseplant/eval_s2` (возобновляемо).
5. Чекпоинт: судья по plant_dev ∪ garden_dev не хуже базы **и** general_dev не падает больше чем на **5 п.п.**

Не ставьте голый `pip install -U transformers trl peft` — ломает API и даёт `HybridCache` ImportError (peft 0.17 × transformers 5).

Локальная проверка пинов (Python ≥3.10):

```bash
python3.12 -m venv .venv-colab && source .venv-colab/bin/activate
pip install -r requirements-colab.txt pytest
python scripts/smoke_colab_stack.py
pytest -q
```

Веса и большие артефакты в git не кладём — ссылка на Drive/HF в отчёте.

## Результаты РК1

Отчёт: `reports/rk1_report.md`. Сводка метрик, генерации, оценки судьи, лог: `results/eval_s1/` (`summary.md`), графики: `reports/plots/`. Выполненные ноутбуки T4: `results/*.ipynb`.

SFT v2 (данные, отобранные судьёй; часть III отчёта): сводка и генерации — `results/eval_s2/`, графики — `reports/plots_s2/`. Итог тот же: ни один чекпоинт не прошёл правило, выбрана база.

LoRA-адаптер S1 (safetensors, + checkpoint-50/93) и копия результатов оценки: [Google Drive: rk1_houseplant](https://drive.google.com/drive/folders/1g7BVgQWBTlDQSJo2VaVZyYB7fwkrT-6c?usp=sharing).

Воспроизвести оценку РК1 без обучения: коммит `efbe34a`, папка адаптера в `MyDrive/rk1_houseplant/s1_plant_qlora`, `notebooks/rk1_eval_overnight.ipynb` в Colab (T4), Secret `OPENROUTER_API_KEY`, Run all.

## Что уже в репозитории

| Артефакт | Путь |
|---|---|
| Манифест | `AGENTS.md` |
| Plant splits | `data/processed/plant_{train,dev,test}.jsonl` |
| General suite | `data/processed/general_{dev,test}.jsonl` |
| Автопроверка | `src/eval/checkers.py` + `tests/test_checkers.py` |
| Judge | `src/eval/openrouter_judge.py`, `configs/judge.yaml` |
| Train | `src/train/sft_qlora.py`, `scripts/train_qlora.py` |
| Журнал | `reports/experiment_log.md` |

## Лицензии данных

Посты Gardening Stack Exchange — CC BY-SA (версии в метаданных постов). Сохраняйте атрибуцию из `data/manifests/plant_metadata.jsonl`.
