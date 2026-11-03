from src.eval.checkers import check_three_numbered_steps


def test_three_steps_pass():
    text = """Here is a plan:
1. Raise humidity with a tray of pebbles and water.
2. Trim only fully brown tips with clean scissors.
3. Check that the pot drains freely after watering.
"""
    r = check_three_numbered_steps(text)
    assert r.passed, r.detail


def test_three_steps_fail_two_only():
    text = """1. Water less often.
2. Move away from the heater.
"""
    r = check_three_numbered_steps(text)
    assert not r.passed


def test_three_steps_fail_duplicate():
    text = """1. Water thoroughly until it drains.
2. Water thoroughly until it drains.
3. Water thoroughly until it drains.
"""
    r = check_three_numbered_steps(text)
    assert not r.passed
    assert "duplicat" in r.detail


def test_three_steps_fail_five_steps():
    text = """1. Check soil moisture with a finger.
2. Empty the saucer after watering.
3. Move the pot closer to the window.
4. Repot into a fast-draining mix.
5. Fertilize lightly in spring.
"""
    r = check_three_numbered_steps(text)
    assert not r.passed


def test_three_steps_fail_empty_third():
    text = """1. Check soil moisture with a finger.
2. Empty the saucer after watering.
3. 
"""
    r = check_three_numbered_steps(text)
    assert not r.passed
