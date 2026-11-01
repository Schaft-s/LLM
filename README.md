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

1. Залейте репозиторий на Drive или сделайте `git clone`.
2. Откройте `notebooks/rk1_colab.ipynb`, Runtime = GPU T4.
3. Установите зависимости из ячейки.
4. Pilot: `python scripts/train_qlora.py --pilot` (save/reload).
5. S1: `python scripts/train_qlora.py`.
6. Сравните base vs adapter на `plant_*` и `general_*` одинаковыми decoding settings.
7. Чекпоинт: plant_dev не хуже базы **и** general_dev не падает больше чем на **5 п.п.**

Веса и большие артефакты в git не кладём — ссылка на Drive/HF в отчёте.

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
