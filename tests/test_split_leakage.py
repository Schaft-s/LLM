import pytest

from src.data.split import assert_no_group_leakage, split_groups


def _row(gid: str, i: int) -> dict:
    return {
        "id": f"{gid}-{i}",
        "group_id": gid,
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
    }


def test_no_leakage():
    rows = [_row(f"g{i}", 0) for i in range(20)]
    splits = split_groups(rows, seed=0, dev_size=3, test_size=3)
    assert_no_group_leakage(splits)
    assert len(splits["plant_dev"]) == 3
    assert len(splits["plant_test"]) == 3


def test_detects_leakage():
    splits = {
        "plant_train": [_row("g1", 0)],
        "plant_dev": [_row("g1", 1)],
        "plant_test": [_row("g2", 0)],
    }
    with pytest.raises(AssertionError):
        assert_no_group_leakage(splits)
