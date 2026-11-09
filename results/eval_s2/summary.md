# RK1 eval summary

Selected: **base**


## dev

| variant | general acc | no invented exp | judge plant | judge garden | judge all (0-8) | critical err | c→w | w→c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.800 | 1.00 | 6.67 | 6.78 | 6.72 | 0.01 | - | - |
| sft_final | 0.863 | 0.96 | 3.90 | 4.28 | 4.09 | 0.16 | 2 | 7 |
| sft_step50 | 0.863 | 0.91 | 3.23 | 3.95 | 3.59 | 0.19 | 2 | 7 |
| sft_step100 | 0.863 | 0.96 | 3.56 | 4.22 | 3.90 | 0.22 | 2 | 7 |

Per-category general accuracy:

| variant | arc_easy | arith | extract | json_schema | logic | stable_facts | units |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.95 | 0.70 | 1.00 | 1.00 | 0.60 | 0.80 | 0.40 |
| sft_final | 0.95 | 0.70 | 1.00 | 1.00 | 0.70 | 1.00 | 0.60 |
| sft_step50 | 0.95 | 0.70 | 1.00 | 1.00 | 0.70 | 1.00 | 0.60 |
| sft_step100 | 0.95 | 0.70 | 1.00 | 1.00 | 0.70 | 1.00 | 0.60 |

## test

| variant | general acc | no invented exp | judge plant | judge garden | judge all (0-8) | critical err | c→w | w→c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.819 | 1.00 | 6.64 | 6.92 | 6.78 | 0.01 | - | - |
| sft_final | 0.850 | 0.97 | 3.97 | 3.92 | 3.95 | 0.14 | 4 | 9 |

Per-category general accuracy:

| variant | arc_easy | arith | extract | json_schema | logic | stable_facts | units |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.93 | 0.95 | 1.00 | 1.00 | 0.45 | 0.85 | 0.45 |
| sft_final | 0.95 | 0.85 | 1.00 | 1.00 | 0.60 | 0.85 | 0.60 |

- rule check `sft_final`: pass=False {"general_drop_pp": -6.25, "general_ok": true, "plant_ok": false, "plant_basis": "judge_mean_total_plant+garden"}

- rule check `sft_step50`: pass=False {"general_drop_pp": -6.25, "general_ok": true, "plant_ok": false, "plant_basis": "judge_mean_total_plant+garden"}

- rule check `sft_step100`: pass=False {"general_drop_pp": -6.25, "general_ok": true, "plant_ok": false, "plant_basis": "judge_mean_total_plant+garden"}
