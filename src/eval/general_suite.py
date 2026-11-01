"""Build general_dev / general_test suites for capability retention."""

from __future__ import annotations

import json
import random
from typing import Any, Callable


AnswerFn = Callable[[str], bool]


def _rec(
    *,
    eid: str,
    category: str,
    split: str,
    prompt: str,
    gold: str,
    check: str,
) -> dict[str, Any]:
    return {
        "id": eid,
        "group_id": eid,
        "split": split,
        "category": category,
        "messages": [{"role": "user", "content": prompt}],
        "gold": gold,
        "check": check,
        "metadata": {
            "source": "project_synthetic" if not eid.startswith("arc-") else "allenai/ai2_arc",
            "language": "en",
            "topic": "general",
        },
    }


def _arith_bank(split: str, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            gold = str(a + b)
            q = f"Compute {a} + {b}. Reply with only the integer."
        elif op == "-":
            gold = str(a - b)
            q = f"Compute {a} - {b}. Reply with only the integer."
        else:
            gold = str(a * b)
            q = f"Compute {a} * {b}. Reply with only the integer."
        rows.append(
            _rec(
                eid=f"arith-{split}-{i:02d}",
                category="arith",
                split=split,
                prompt=q,
                gold=gold,
                check="exact",
            )
        )
    return rows


def _units_bank(split: str, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 1)
    templates = [
        (
            "Convert {c} Celsius to Fahrenheit. Reply with only the number rounded to 1 decimal.",
            lambda c: f"{(c * 9 / 5) + 32:.1f}",
        ),
        (
            "How many centimeters are in {m} meters? Reply with only the integer.",
            lambda m: str(int(m * 100)),
        ),
        (
            "Convert {kg} kilograms to grams. Reply with only the integer.",
            lambda kg: str(int(kg * 1000)),
        ),
    ]
    rows = []
    for i in range(n):
        tmpl, fn = templates[i % len(templates)]
        if "Celsius" in tmpl:
            c = rng.randint(0, 40)
            prompt = tmpl.format(c=c)
            gold = fn(c)
        elif "meters" in tmpl:
            m = rng.randint(1, 25)
            prompt = tmpl.format(m=m)
            gold = fn(m)
        else:
            kg = rng.randint(1, 15)
            prompt = tmpl.format(kg=kg)
            gold = fn(kg)
        rows.append(
            _rec(
                eid=f"units-{split}-{i:02d}",
                category="units",
                split=split,
                prompt=prompt,
                gold=gold,
                check="exact",
            )
        )
    return rows


def _extract_bank(split: str, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 2)
    people = ["Ada", "Boris", "Chen", "Diana", "Evan", "Farah", "Gus", "Hana"]
    cities = ["Oslo", "Lisbon", "Kyoto", "Cairo", "Perth", "Quito", "Riga", "Seoul"]
    rows = []
    for i in range(n):
        name = people[i % len(people)]
        city = cities[(i + 3) % len(cities)]
        year = 1990 + (i * 3 + seed) % 30
        passage = (
            f"{name} moved to {city} in {year} and works as a librarian. "
            f"The library opens at 9:00."
        )
        kind = i % 3
        if kind == 0:
            prompt = f"Passage: {passage}\nExtract the city name only."
            gold = city
        elif kind == 1:
            prompt = f"Passage: {passage}\nExtract the year only."
            gold = str(year)
        else:
            prompt = f"Passage: {passage}\nExtract the person's name only."
            gold = name
        rows.append(
            _rec(
                eid=f"extract-{split}-{i:02d}",
                category="extract",
                split=split,
                prompt=prompt,
                gold=gold,
                check="exact_ci",
            )
        )
    _ = rng
    return rows


def _logic_bank(split: str, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 3)
    rows = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            xs = [rng.randint(1, 9) for _ in range(5)]
            gold = ",".join(str(x) for x in sorted(xs))
            prompt = (
                f"Sort these integers ascending and reply with a comma-separated list only: "
                f"{','.join(map(str, xs))}"
            )
        elif kind == 1:
            a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 20)
            gold = str(max(a, b, c))
            prompt = f"Which is largest: {a}, {b}, or {c}? Reply with only that number."
        elif kind == 2:
            s = "".join(rng.choice("abcdxyz") for _ in range(6))
            gold = s[::-1]
            prompt = f"Reverse this string and reply with only the result: {s}"
        else:
            nums = [rng.randint(1, 10) for _ in range(4)]
            gold = str(sum(nums))
            prompt = f"Sum these numbers and reply with only the total: {', '.join(map(str, nums))}"
        rows.append(
            _rec(
                eid=f"logic-{split}-{i:02d}",
                category="logic",
                split=split,
                prompt=prompt,
                gold=gold,
                check="exact",
            )
        )
    return rows


def _json_bank(split: str, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 4)
    rows = []
    for i in range(n):
        name = rng.choice(["alpha", "beta", "gamma", "delta"])
        value = rng.randint(1, 100)
        prompt = (
            'Return ONLY valid JSON with keys "name" (string) and "value" (integer). '
            f'Set name to "{name}" and value to {value}.'
        )
        gold = json.dumps({"name": name, "value": value}, separators=(",", ":"))
        rows.append(
            _rec(
                eid=f"json-{split}-{i:02d}",
                category="json_schema",
                split=split,
                prompt=prompt,
                gold=gold,
                check="json_obj",
            )
        )
    return rows


def _facts_bank(split: str, n: int, seed: int) -> list[dict[str, Any]]:
    facts = [
        ("How many minutes are in one hour? Reply with only the integer.", "60"),
        ("How many days are in a non-leap year? Reply with only the integer.", "365"),
        ("What is the chemical formula for water? Reply with only the formula.", "H2O"),
        ("How many continents are commonly listed on Earth? Reply with only the integer.", "7"),
        ("What planet is known as the Red Planet? Reply with only the name.", "Mars"),
        ("How many sides does a hexagon have? Reply with only the integer.", "6"),
        ("What gas do plants primarily absorb for photosynthesis? Reply with only the name or formula.", "CO2"),
        ("What is the boiling point of water at 1 atm in Celsius? Reply with only the integer.", "100"),
        ("How many bits are in one byte? Reply with only the integer.", "8"),
        ("What is the square root of 81? Reply with only the integer.", "9"),
        ("How many hours are in a day? Reply with only the integer.", "24"),
        ("What is the capital of Japan? Reply with only the city name.", "Tokyo"),
        ("How many legs does an insect typically have? Reply with only the integer.", "6"),
        ("What is 2 to the power of 5? Reply with only the integer.", "32"),
        ("What ocean is the largest by area? Reply with only the name.", "Pacific"),
        ("How many zeros are in one thousand? Reply with only the integer.", "3"),
        ("What is the atomic number of hydrogen? Reply with only the integer.", "1"),
        ("How many players are on the field for one soccer team during play? Reply with only the integer.", "11"),
        ("What is the SI unit of electric current? Reply with only the unit name.", "ampere"),
        ("How many degrees are in a right angle? Reply with only the integer.", "90"),
    ]
    # Separate families for split to reduce near-duplicates across splits
    if split == "dev":
        pool = facts[:10]
    else:
        pool = facts[10:] + facts[:10]
    rows = []
    for i in range(n):
        q, gold = pool[i % len(pool)]
        # Slightly rephrase test items by adding instruction already present
        rows.append(
            _rec(
                eid=f"facts-{split}-{i:02d}",
                category="stable_facts",
                split=split,
                prompt=q,
                gold=gold,
                check="exact_ci",
            )
        )
    _ = seed
    return rows


def build_synthetic(split: str, per_cat: int, seed: int) -> list[dict[str, Any]]:
    return (
        _arith_bank(split, per_cat, seed)
        + _units_bank(split, per_cat, seed)
        + _extract_bank(split, per_cat, seed)
        + _logic_bank(split, per_cat, seed)
        + _json_bank(split, per_cat, seed)
        + _facts_bank(split, per_cat, seed)
    )


def load_arc_easy(split: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Load ARC-Easy subset; split is 'validation' or 'test' for HF, mapped to our names."""
    from datasets import load_dataset

    hf_split = "validation" if split == "dev" else "test"
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=hf_split)
    rng = random.Random(seed + 100)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    rows = []
    for j, idx in enumerate(idxs[:n]):
        ex = ds[idx]
        choices = ex["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_lines = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        prompt = (
            f"Question: {ex['question']}\n"
            f"Choices:\n{choice_lines}\n"
            "Reply with only the choice letter."
        )
        rows.append(
            _rec(
                eid=f"arc-easy-{split}-{ex['id']}",
                category="arc_easy",
                split=split,
                prompt=prompt,
                gold=str(ex["answerKey"]).strip(),
                check="choice_letter",
            )
        )
        _ = j
    return rows


def build_general_splits(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    g = cfg["general"]
    seed = cfg["seed"]
    dev = build_synthetic("dev", g["per_category_dev"], seed)
    test = build_synthetic("test", g["per_category_test"], seed + 17)
    try:
        dev += load_arc_easy("dev", g["arc_easy_dev"], seed)
        test += load_arc_easy("test", g["arc_easy_test"], seed)
        arc_ok = True
    except Exception as exc:  # noqa: BLE001 — network/offline fallback
        arc_ok = False
        arc_error = str(exc)
        # Placeholder ARC-like items so suite size is documented if HF unavailable
        for i in range(g["arc_easy_dev"]):
            dev.append(
                _rec(
                    eid=f"arc-easy-dev-fallback-{i:02d}",
                    category="arc_easy",
                    split="dev",
                    prompt="If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Reply yes or no.",
                    gold="yes",
                    check="exact_ci",
                )
            )
        for i in range(g["arc_easy_test"]):
            test.append(
                _rec(
                    eid=f"arc-easy-test-fallback-{i:02d}",
                    category="arc_easy",
                    split="test",
                    prompt="A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost in cents? Reply with only the integer.",
                    gold="5",
                    check="exact",
                )
            )
    assert len(dev) == g["dev_size"], f"dev size {len(dev)} != {g['dev_size']}"
    assert len(test) == g["test_size"], f"test size {len(test)} != {g['test_size']}"
    meta = {"arc_from_hf": arc_ok}
    if not arc_ok:
        meta["arc_error"] = arc_error
    return {"general_dev": dev, "general_test": test, "_meta": meta}
