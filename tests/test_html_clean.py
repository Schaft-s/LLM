from src.data.html_clean import has_img_markup, html_to_text
from src.data.parse_se import parse_tags


def test_parse_pipe_tags():
    assert parse_tags("|houseplants|sunlight|") == ["houseplants", "sunlight"]


def test_html_lists_and_img():
    html = "<p>Hello</p><ul><li>One</li><li>Two</li></ul><img src='x.png'/>"
    text = html_to_text(html)
    assert "Hello" in text
    assert "- One" in text
    assert has_img_markup(html)
