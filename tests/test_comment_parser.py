import pytest

from comment_parser import CommentParser


@pytest.fixture
def parser():
    return CommentParser()


def test_shorter_returns_true(parser):
    assert parser.is_shorter_than("hello world", 5) is True


def test_equal_length_returns_false(parser):
    assert parser.is_shorter_than("one two three", 3) is False


def test_longer_returns_false(parser):
    assert parser.is_shorter_than("one two three four five", 3) is False


def test_empty_comment_returns_true(parser):
    assert parser.is_shorter_than("", 1) is True


def test_extra_whitespace_counts_correctly(parser):
    # split() collapses runs of whitespace, so "a  b" is 2 words
    assert parser.is_shorter_than("a  b", 3) is True


def test_single_word_shorter(parser):
    assert parser.is_shorter_than("hello", 2) is True


def test_single_word_not_shorter(parser):
    assert parser.is_shorter_than("hello", 1) is False
