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

Сырой dump: `data/raw/` (gitignored). SHA1 архива: `718db7040d0b608b1c969d4630d8bb95db87eec5`.

## Colab T4

1. Откройте `notebooks/rk1_colab.ipynb`, Runtime = GPU T4.
2. Ячейка clone/`git pull`, затем `bash scripts/colab_install.sh` (пины 4.57/trl/peft; без `--force-reinstall`).
3. Pilot: `python scripts/train_qlora.py --pilot` (save/reload + 1 generation).
4. S1: `python scripts/train_qlora.py`.
5. Сравните base vs adapter на `plant_*` и `general_*` одинаковыми decoding settings: `notebooks/rk1_eval_overnight.ipynb` (адаптер с Drive, результаты на Drive, возобновляемо).
6. Чекпоинт: plant_dev не хуже базы **и** general_dev не падает больше чем на **5 п.п.**

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

LoRA-адаптер S1 (safetensors, + checkpoint-50/93) и копия результатов оценки: [Google Drive: rk1_houseplant](https://drive.google.com/drive/folders/1g7BVgQWBTlDQSJo2VaVZyYB7fwkrT-6c?usp=sharing).

Воспроизвести оценку без обучения: загрузить папку адаптера в `MyDrive/rk1_houseplant/s1_plant_qlora`, открыть `notebooks/rk1_eval_overnight.ipynb` в Colab (T4), добавить Secret `OPENROUTER_API_KEY`, Run all.

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
