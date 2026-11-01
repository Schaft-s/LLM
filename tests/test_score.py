from src.eval.score import score_answer


def test_exact():
    assert score_answer("42", "42", "exact")
    assert score_answer("42\nexplanation", "42", "exact")


def test_json_obj():
    gold = '{"name":"alpha","value":3}'
    assert score_answer('{"name":"alpha","value":3}', gold, "json_obj")
    assert score_answer('Here you go:\n{"name":"alpha","value":3}\n', gold, "json_obj")
    assert not score_answer('{"name":"alpha","value":4}', gold, "json_obj")


def test_choice_letter():
    assert score_answer("Answer: B", "B", "choice_letter")
    assert not score_answer("I like bananas and C is wrong", "A", "choice_letter")
