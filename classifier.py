"""Loads and runs a fine-tuned BERT reaction classifier."""

from transformers import BertForSequenceClassification, pipeline


class ReactionClassifier:
    """Wraps a fine-tuned BertForSequenceClassification model for comment classification."""

    def __init__(self) -> None:
        self._pipeline = None  # set by load_classifier(); None means not yet loaded

    def load_classifier(self, model_path: str) -> None:
        """Load the model and initialise the text-classification pipeline.

        Args:
            model_path: Path to the directory containing the model weights,
                tokenizer config, and vocab files (e.g. ``"model_output"``).
        """
        # from_pretrained reads config.json + model.safetensors from model_path
        model = BertForSequenceClassification.from_pretrained(model_path)
        # passing tokenizer as a path lets the pipeline load it from the same directory
        self._pipeline = pipeline("text-classification", model=model, tokenizer=model_path)

    def classify_comment(self, comment: str) -> dict:
        """Classify a single comment and return the top label with its confidence score.

        Args:
            comment: The raw text comment to classify.

        Returns:
            A dict with keys ``"label"`` (str) and ``"score"`` (float), e.g.
            ``{"label": "ja", "score": 0.98}``.

        Raises:
            RuntimeError: If called before :meth:`load_classifier`.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier not loaded. Call load_classifier() first.")
        result = self._pipeline(comment)
        return result[0]  # pipeline returns a list; we always classify one comment
