# Автопроверка: 3 numbered steps

**Правило:** если промпт просит ровно три нумерованных шага, ответ должен содержать три непустых различных пункта `1./2./3.`.

## Pass

```
1. Raise humidity with a pebble tray.
2. Trim only fully brown tips with clean scissors.
3. Confirm the pot drains freely after watering.
```

## Fail — только два пункта

```
1. Water less often.
2. Move away from the heater.
```

## Fail — дублирование

```
1. Water thoroughly until it drains.
2. Water thoroughly until it drains.
3. Water thoroughly until it drains.
```
