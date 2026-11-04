# RK1 eval summary

Selected: **base**


## dev

| variant | general acc | format pass | judge mean (0-8) | critical err | c→w | w→c |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.825 | 0.92 | 6.54 | 0.03 | - | - |
| s1_final | 0.863 | 0.18 | 3.15 | 0.28 | 3 | 6 |
| s1_step50 | 0.863 | 0.23 | 2.97 | 0.28 | 3 | 6 |

Per-category general accuracy:

| variant | arc_easy | arith | extract | json_schema | logic | stable_facts | units |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.95 | 0.90 | 1.00 | 1.00 | 0.60 | 0.80 | 0.40 |
| s1_final | 1.00 | 0.70 | 1.00 | 1.00 | 0.60 | 1.00 | 0.60 |
| s1_step50 | 1.00 | 0.70 | 1.00 | 1.00 | 0.60 | 1.00 | 0.60 |

## test

| variant | general acc | format pass | judge mean (0-8) | critical err | c→w | w→c |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.819 | 0.97 | 6.64 | 0.00 | - | - |
| s1_final | 0.863 | 0.13 | 3.67 | 0.13 | 4 | 11 |

Per-category general accuracy:

| variant | arc_easy | arith | extract | json_schema | logic | stable_facts | units |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.90 | 1.00 | 1.00 | 1.00 | 0.50 | 0.85 | 0.40 |
| s1_final | 0.95 | 0.95 | 1.00 | 1.00 | 0.55 | 0.85 | 0.65 |

- rule check `s1_final`: pass=False {"general_drop_pp": -3.750000000000009, "general_ok": true, "plant_ok": false, "plant_basis": "judge_mean_total"}

- rule check `s1_step50`: pass=False {"general_drop_pp": -3.750000000000009, "general_ok": true, "plant_ok": false, "plant_basis": "judge_mean_total"}
