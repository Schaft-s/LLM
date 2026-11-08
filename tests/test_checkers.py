import pytest

from src.eval.checkers import check_no_invented_experience, check_three_numbered_steps


@pytest.mark.parametrize(
    "text",
    [
        "Ferns prefer evenly moist soil; brown tips usually mean low humidity.",
        "I think the roots are rotting. I'd use a gritty mix and water less.",
        "### What Product Should I Use?\nA plastic divider works well.",
        "I would suggest checking the drainage holes first.",
    ],
)
def test_no_invented_experience_pass(text):
    r = check_no_invented_experience(text)
    assert r.passed, r.detail


@pytest.mark.parametrize(
    "text",
    [
        # real S1 output (scale / submersion question)
        "I have not heard of this method before, but I have seen it done with other plants.",
        "I have used it with succulents and it works very well.",
        "I've never seen a pot of soil that was completely sterile.",
        "In my experience, orchids rebloom after a cool night period.",
        "My own orchids sit in an east window.",
    ],
)
def test_no_invented_experience_fail(text):
    r = check_no_invented_experience(text)
    assert not r.passed
    assert "claims experience" in r.detail


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
