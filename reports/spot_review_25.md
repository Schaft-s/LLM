# Spot-review 25 (2026-11-01)

Просмотрена выборка `data/manifests/spot_review_25.jsonl` после автофильтра. Цель — оценка yield и ложных срабатываний, не полный ручной корпус.

## Итог

| Решение | Число |
|---|---:|
| keep (согласен с auto) | 16 |
| drop (согласен с auto) | 4 |
| flip keep→drop (вне темы / слабый fit) | 4 |
| flip drop→keep (эвристика ошиблась) | 1 |

Автофильтр в целом адекватен: технические ~839 совпали с аудитом (838). Sufficiency срезала мало (~14). Основные потери — `<img>` (~1935).

## Flips

| id | Было | Стало | Причина |
|---|---|---|---|
| gardening-q20502-a20503 | keep | drop | Уличный 20ft Ficus, не комнатный уход |
| gardening-q17501-a17675 | keep | drop | Овощи на балконе; вне декоративных houseplants |
| gardening-q22777-a22778 | keep | drop | Овощные контейнеры / safety lumber |
| gardening-q53109-a53110 | keep | drop | Уличные planter boxes на бетоне |
| gardening-q37771-a37773 | drop | keep | Не identification: вопрос про размножение при root rot; текст самодостаточен |

## Ложные эвристики

- `identification_only` иногда цепляет care-вопросы с словом «identify» в другом смысле / диагнозе.
- `outdoor_agriculture` на orchid-mounting ложно сработало через слово `farm`? (в этом сэмпле — bucket outdoor_agriculture на q17397; формулировка про rural materials — спорно; для первого прогона оставлен drop).

## Вывод для корпуса

Ожидаемая доля off-topic среди auto-keep порядка **~15–20%** по этому сэмплу keep-бакета. Для РК1 оставляем автокорпус 825 и фиксируем ограничение в отчёте; точечный denylist по flip-ID можно применить перед S1.

Denylist к исключению из train/dev/test перед обучением:

- `gardening-q20502-a20503`
- `gardening-q17501-a17675`
- `gardening-q22777-a22778`
- `gardening-q53109-a53110`
