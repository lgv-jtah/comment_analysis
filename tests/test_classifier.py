from unittest.mock import MagicMock, patch

import pytest

from classifier import ReactionClassifier


@pytest.fixture
def classifier():
    return ReactionClassifier()


def test_load_classifier_sets_pipeline(classifier):
    mock_model = MagicMock()
    mock_pipe = MagicMock()

    with (
        patch("classifier.BertForSequenceClassification.from_pretrained", return_value=mock_model) as mock_from_pretrained,
        patch("classifier.pipeline", return_value=mock_pipe) as mock_pipeline,
    ):
        classifier.load_classifier("model_output")

    mock_from_pretrained.assert_called_once_with("model_output")
    mock_pipeline.assert_called_once_with("text-classification", model=mock_model, tokenizer="model_output")
    assert classifier._pipeline is mock_pipe


def test_classify_comment_returns_result(classifier):
    mock_pipe = MagicMock(return_value=[{"label": "ja", "score": 0.98}])
    classifier._pipeline = mock_pipe

    result = classifier.classify_comment("Das ist ein toller Kommentar.")

    mock_pipe.assert_called_once_with("Das ist ein toller Kommentar.")
    assert result == {"label": "ja", "score": 0.98}


def test_classify_comment_raises_when_not_loaded(classifier):
    with pytest.raises(RuntimeError, match="load_classifier"):
        classifier.classify_comment("some comment")


def test_classify_comment_returns_first_result(classifier):
    classifier._pipeline = MagicMock(return_value=[{"label": "nein", "score": 0.75}, {"label": "ja", "score": 0.25}])

    result = classifier.classify_comment("Ein Kommentar.")

    assert result["label"] == "nein"
    assert result["score"] == 0.75
