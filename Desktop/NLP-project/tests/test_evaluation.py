"""Tests for src/evaluation.py - Comprehensive evaluation suite."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.evaluation import (
    BertScoreEvaluator,
    ComprehensiveEvaluator,
    CoverageEvaluator,
    FaithfulnessEvaluator,
    RedundancyEvaluator,
    RougeEvaluator,
)

# ===== Fixtures =====


@pytest.fixture
def sample_predictions():
    """Sample prediction summaries for testing."""
    return [
        "This is a test summary with some content.",
        "Another summary with different words and phrases.",
        "A third summary for testing purposes.",
    ]


@pytest.fixture
def sample_references():
    """Sample reference summaries for testing."""
    return [
        "This is the reference summary content.",
        "Reference summary with similar content.",
        "Third reference summary for comparison.",
    ]


@pytest.fixture
def sample_sources():
    """Sample source documents for testing."""
    return [
        "This is the full source document with lots of content that "
        "needs to be summarized into a shorter form.",
        "Another source document with multiple sentences. It contains "
        "various information. This should be condensed.",
        "A third source document for testing coverage and faithfulness "
        "metrics with detailed content.",
    ]


@pytest.fixture
def single_prediction():
    """Single prediction for pair testing."""
    return "This is a summary of the document."


@pytest.fixture
def single_reference():
    """Single reference for pair testing."""
    return "This is the reference summary."


@pytest.fixture
def single_source():
    """Single source for pair testing."""
    return (
        "This is the source document with multiple sentences. "
        "It contains information to be summarized."
    )


# ===== RougeEvaluator Tests =====


class TestRougeEvaluator:
    """Tests for RougeEvaluator class."""

    def test_rouge_initialization_default(self):
        """Test RougeEvaluator initialization with default parameters."""
        evaluator = RougeEvaluator()
        assert evaluator.rouge_types == ["rouge1", "rouge2", "rougeL"]
        assert evaluator.scorer is not None

    def test_rouge_initialization_custom(self):
        """Test RougeEvaluator initialization with custom rouge types."""
        custom_types = ["rouge1", "rougeL"]
        evaluator = RougeEvaluator(rouge_types=custom_types)
        assert evaluator.rouge_types == custom_types

    def test_rouge_evaluate_single_pair(self, sample_predictions, sample_references):
        """Test ROUGE evaluation with single prediction-reference pair."""
        evaluator = RougeEvaluator()
        results = evaluator.evaluate([sample_predictions[0]], [sample_references[0]])

        # Check that all expected keys are present
        assert "rouge1_precision" in results
        assert "rouge1_recall" in results
        assert "rouge1_fmeasure" in results
        assert "rouge2_precision" in results
        assert "rouge2_recall" in results
        assert "rouge2_fmeasure" in results
        assert "rougeL_precision" in results
        assert "rougeL_recall" in results
        assert "rougeL_fmeasure" in results

    def test_rouge_evaluate_batch(self, sample_predictions, sample_references):
        """Test ROUGE evaluation with batch of predictions."""
        evaluator = RougeEvaluator()
        results = evaluator.evaluate(sample_predictions, sample_references)

        # Check all metrics are present
        assert len(results) == 9  # 3 rouge types × 3 metrics

        # All scores should be between 0 and 1
        for key, value in results.items():
            assert 0.0 <= value <= 1.0, f"{key} score {value} not in [0, 1]"

    def test_rouge_score_ranges(self, sample_predictions, sample_references):
        """Test that ROUGE scores are in valid range [0, 1]."""
        evaluator = RougeEvaluator()
        results = evaluator.evaluate(sample_predictions, sample_references)

        for metric, score in results.items():
            assert isinstance(score, (float, np.floating))
            assert 0.0 <= score <= 1.0

    def test_rouge_empty_predictions(self):
        """Test ROUGE with empty predictions."""
        evaluator = RougeEvaluator()
        results = evaluator.evaluate([""], ["reference"])

        # Should still return valid structure
        assert "rouge1_fmeasure" in results
        assert isinstance(results["rouge1_fmeasure"], (float, np.floating))

    def test_rouge_identical_texts(self):
        """Test ROUGE with identical prediction and reference."""
        evaluator = RougeEvaluator()
        text = "This is a test summary with multiple words."
        results = evaluator.evaluate([text], [text])

        # F-measure should be close to 1.0 for identical texts
        assert results["rouge1_fmeasure"] > 0.9
        assert results["rougeL_fmeasure"] > 0.9


# ===== BertScoreEvaluator Tests =====


class TestBertScoreEvaluator:
    """Tests for BertScoreEvaluator class."""

    def test_bertscore_initialization_default(self):
        """Test BertScoreEvaluator initialization with default model."""
        evaluator = BertScoreEvaluator()
        assert evaluator.model_type == "microsoft/deberta-xlarge-mnli"

    def test_bertscore_initialization_custom(self):
        """Test BertScoreEvaluator initialization with custom model."""
        custom_model = "bert-base-uncased"
        evaluator = BertScoreEvaluator(model_type=custom_model)
        assert evaluator.model_type == custom_model

    @patch("src.evaluation.bert_score")
    def test_bertscore_evaluate_mocked(
        self, mock_bert_score, sample_predictions, sample_references
    ):
        """Test BERTScore evaluation with mocked bert_score function."""
        # Mock the bert_score return values
        mock_bert_score.return_value = (
            torch.tensor([0.92, 0.88, 0.90]),
            torch.tensor([0.85, 0.87, 0.86]),
            torch.tensor([0.88, 0.87, 0.88]),
        )

        evaluator = BertScoreEvaluator()
        results = evaluator.evaluate(sample_predictions, sample_references)

        # Check that bert_score was called
        mock_bert_score.assert_called_once()

        # Check results structure
        assert "bertscore_precision" in results
        assert "bertscore_recall" in results
        assert "bertscore_f1" in results

        # Check values are close to mocked values
        assert abs(results["bertscore_precision"] - 0.90) < 0.01
        assert abs(results["bertscore_recall"] - 0.86) < 0.01
        assert abs(results["bertscore_f1"] - 0.876) < 0.01

    @patch("src.evaluation.bert_score")
    def test_bertscore_output_format(
        self, mock_bert_score, sample_predictions, sample_references
    ):
        """Test BERTScore output format validation."""
        mock_bert_score.return_value = (
            torch.tensor([0.9]),
            torch.tensor([0.85]),
            torch.tensor([0.87]),
        )

        evaluator = BertScoreEvaluator()
        results = evaluator.evaluate(sample_predictions[:1], sample_references[:1])

        # All values should be floats
        assert isinstance(results["bertscore_precision"], float)
        assert isinstance(results["bertscore_recall"], float)
        assert isinstance(results["bertscore_f1"], float)


# ===== FaithfulnessEvaluator Tests =====


class TestFaithfulnessEvaluator:
    """Tests for FaithfulnessEvaluator class."""

    @patch("transformers.pipeline")
    def test_faithfulness_initialization(self, mock_pipeline):
        """Test FaithfulnessEvaluator initialization."""
        mock_nli = MagicMock()
        mock_pipeline.return_value = mock_nli

        evaluator = FaithfulnessEvaluator()

        mock_pipeline.assert_called_once_with(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )
        assert evaluator.nli_model == mock_nli

    @patch("transformers.pipeline")
    @patch("nltk.sent_tokenize")
    def test_faithfulness_evaluate_pair_mocked(self, mock_sent_tokenize, mock_pipeline):
        """Test faithfulness evaluation for single pair with mocked NLI."""
        # Mock sentence tokenization
        mock_sent_tokenize.return_value = ["First sentence.", "Second sentence."]

        # Mock NLI pipeline
        mock_nli = MagicMock()
        mock_nli.return_value = {"scores": [0.95]}
        mock_pipeline.return_value = mock_nli

        evaluator = FaithfulnessEvaluator()
        score = evaluator.evaluate_pair(
            "Source document.", "First sentence. Second sentence."
        )

        # Should average the scores from both sentences
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == 0.95  # Both sentences got 0.95

    @patch("transformers.pipeline")
    @patch("nltk.sent_tokenize")
    def test_faithfulness_empty_summary(self, mock_sent_tokenize, mock_pipeline):
        """Test faithfulness with empty summary."""
        mock_sent_tokenize.return_value = []
        mock_pipeline.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator()
        score = evaluator.evaluate_pair("Source", "")

        assert score == 0.0

    @patch("transformers.pipeline")
    @patch("nltk.sent_tokenize")
    @patch("nltk.download")
    def test_faithfulness_nltk_download_trigger(
        self, mock_download, mock_sent_tokenize, mock_pipeline
    ):
        """Test that NLTK punkt download is triggered on LookupError."""
        # First call raises LookupError, second call succeeds
        mock_sent_tokenize.side_effect = [
            LookupError("punkt not found"),
            ["Sentence one."],
        ]

        mock_nli = MagicMock()
        mock_nli.return_value = {"scores": [0.9]}
        mock_pipeline.return_value = mock_nli

        evaluator = FaithfulnessEvaluator()
        score = evaluator.evaluate_pair("Source", "Sentence one.")

        # Verify download was called
        mock_download.assert_called_once_with("punkt")
        assert isinstance(score, float)

    @patch("transformers.pipeline")
    @patch("nltk.sent_tokenize")
    def test_faithfulness_exception_handling(self, mock_sent_tokenize, mock_pipeline):
        """Test faithfulness exception handling returns 0.5."""
        mock_sent_tokenize.return_value = ["Test sentence."]

        # Mock NLI to raise exception
        mock_nli = MagicMock()
        mock_nli.side_effect = Exception("NLI error")
        mock_pipeline.return_value = mock_nli

        evaluator = FaithfulnessEvaluator()
        score = evaluator.evaluate_pair("Source", "Test sentence.")

        # Should return 0.5 on exception
        assert score == 0.5

    @patch("transformers.pipeline")
    @patch("nltk.sent_tokenize")
    def test_faithfulness_evaluate_batch(self, mock_sent_tokenize, mock_pipeline):
        """Test faithfulness evaluation for batch."""
        mock_sent_tokenize.return_value = ["Sentence."]

        mock_nli = MagicMock()
        mock_nli.return_value = {"scores": [0.85]}
        mock_pipeline.return_value = mock_nli

        evaluator = FaithfulnessEvaluator()

        predictions = ["Summary 1.", "Summary 2."]
        sources = ["Source 1.", "Source 2."]

        with patch("src.evaluation.tqdm", side_effect=lambda x, **kwargs: x):
            results = evaluator.evaluate(predictions, sources)

        assert "faithfulness_mean" in results
        assert "faithfulness_std" in results
        assert results["faithfulness_mean"] == 0.85
        assert results["faithfulness_std"] == 0.0


# ===== CoverageEvaluator Tests =====


class TestCoverageEvaluator:
    """Tests for CoverageEvaluator class."""

    def test_coverage_initialization(self):
        """Test CoverageEvaluator initialization."""
        evaluator = CoverageEvaluator()
        assert evaluator.vectorizer is not None

    def test_coverage_evaluate_pair(self):
        """Test coverage evaluation for single pair."""
        evaluator = CoverageEvaluator()
        source = "The quick brown fox jumps over the lazy dog repeatedly."
        summary = "The fox jumps over the dog."

        score = evaluator.evaluate_pair(source, summary)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_coverage_empty_source(self):
        """Test coverage with empty source."""
        evaluator = CoverageEvaluator()
        score = evaluator.evaluate_pair("", "Summary text")

        assert score == 0.0

    def test_coverage_exception_handling(self):
        """Test coverage exception handling."""
        evaluator = CoverageEvaluator()

        # Patch vectorizer to raise exception
        with patch.object(
            evaluator.vectorizer, "fit_transform", side_effect=Exception("Error")
        ):
            score = evaluator.evaluate_pair("Source", "Summary")
            assert score == 0.0

    def test_coverage_evaluate_batch(self, sample_predictions, sample_sources):
        """Test coverage evaluation for batch."""
        evaluator = CoverageEvaluator()
        results = evaluator.evaluate(sample_predictions, sample_sources)

        assert "coverage_mean" in results
        assert "coverage_std" in results
        assert isinstance(results["coverage_mean"], (float, np.floating))
        assert isinstance(results["coverage_std"], (float, np.floating))

    def test_coverage_perfect_overlap(self):
        """Test coverage with perfect summary (all important words)."""
        evaluator = CoverageEvaluator()
        text = "important document with critical information"

        # Summary contains all important words
        score = evaluator.evaluate_pair(text, text)

        # Should have high coverage
        assert score > 0.5


# ===== RedundancyEvaluator Tests =====


class TestRedundancyEvaluator:
    """Tests for RedundancyEvaluator class."""

    def test_redundancy_initialization(self):
        """Test RedundancyEvaluator doesn't require initialization params."""
        evaluator = RedundancyEvaluator()
        assert evaluator is not None

    @patch("nltk.sent_tokenize")
    def test_redundancy_single_sentence(self, mock_sent_tokenize):
        """Test redundancy with single sentence (no redundancy)."""
        mock_sent_tokenize.return_value = ["This is a single sentence."]

        evaluator = RedundancyEvaluator()
        score = evaluator.evaluate_pair("This is a single sentence.")

        assert score == 0.0

    @patch("nltk.sent_tokenize")
    def test_redundancy_repeated_ngrams(self, mock_sent_tokenize):
        """Test redundancy detection with repeated n-grams."""
        mock_sent_tokenize.return_value = [
            "The cat sat on the mat.",
            "The cat sat on the floor.",
        ]

        evaluator = RedundancyEvaluator()
        score = evaluator.evaluate_pair(
            "The cat sat on the mat. The cat sat on the floor."
        )

        # Should detect some redundancy due to repeated "the cat sat"
        assert score >= 0.0
        assert isinstance(score, float)

    @patch("nltk.sent_tokenize")
    @patch("nltk.download")
    def test_redundancy_nltk_download(self, mock_download, mock_sent_tokenize):
        """Test NLTK download trigger in redundancy evaluation."""
        mock_sent_tokenize.side_effect = [
            LookupError("punkt not found"),
            ["Sentence."],
        ]

        evaluator = RedundancyEvaluator()
        evaluator.evaluate_pair("Sentence.")

        mock_download.assert_called_once_with("punkt")

    @patch("nltk.sent_tokenize")
    def test_redundancy_empty_ngrams(self, mock_sent_tokenize):
        """Test redundancy with text that produces no n-grams."""
        mock_sent_tokenize.return_value = ["a b", "c d"]

        evaluator = RedundancyEvaluator()
        score = evaluator.evaluate_pair("a b. c d.")

        # Should handle gracefully
        assert isinstance(score, float)
        assert score >= 0.0

    @patch("nltk.sent_tokenize")
    def test_redundancy_evaluate_batch(self, mock_sent_tokenize, sample_predictions):
        """Test redundancy evaluation for batch."""
        mock_sent_tokenize.return_value = ["Sentence one.", "Sentence two."]

        evaluator = RedundancyEvaluator()
        results = evaluator.evaluate(sample_predictions)

        assert "redundancy_mean" in results
        assert "redundancy_std" in results
        assert isinstance(results["redundancy_mean"], (float, np.floating))
        assert isinstance(results["redundancy_std"], (float, np.floating))


# ===== ComprehensiveEvaluator Tests =====


class TestComprehensiveEvaluator:
    """Tests for ComprehensiveEvaluator class."""

    def test_comprehensive_init_default_metrics(self):
        """Test ComprehensiveEvaluator initialization with default metrics."""
        with patch("src.evaluation.BertScoreEvaluator"):
            evaluator = ComprehensiveEvaluator()

            assert "rouge" in evaluator.metrics
            assert "bertscore" in evaluator.metrics
            assert "coverage" in evaluator.metrics
            assert "redundancy" in evaluator.metrics

    def test_comprehensive_init_custom_metrics(self):
        """Test ComprehensiveEvaluator initialization with custom metrics."""
        custom_metrics = ["rouge", "coverage"]
        evaluator = ComprehensiveEvaluator(metrics=custom_metrics)

        assert evaluator.metrics == custom_metrics
        assert "rouge" in evaluator.evaluators
        assert "coverage" in evaluator.evaluators
        assert "bertscore" not in evaluator.evaluators

    @patch("builtins.print")
    def test_comprehensive_evaluate_rouge_only(
        self, mock_print, sample_predictions, sample_references
    ):
        """Test comprehensive evaluation with ROUGE only."""
        evaluator = ComprehensiveEvaluator(metrics=["rouge"])
        results = evaluator.evaluate(sample_predictions, sample_references)

        # Should have ROUGE metrics
        assert "rouge1_fmeasure" in results
        assert "rouge2_fmeasure" in results

        # Should have metadata
        assert "num_samples" in results
        assert "avg_prediction_length" in results
        assert "avg_reference_length" in results

        assert results["num_samples"] == 3

    @patch("src.evaluation.bert_score")
    @patch("builtins.print")
    def test_comprehensive_evaluate_all_metrics_mocked(
        self,
        mock_print,
        mock_bert_score,
        sample_predictions,
        sample_references,
        sample_sources,
    ):
        """Test comprehensive evaluation with all metrics using mocks."""
        # Mock BERTScore
        mock_bert_score.return_value = (
            torch.tensor([0.9, 0.9, 0.9]),
            torch.tensor([0.85, 0.85, 0.85]),
            torch.tensor([0.87, 0.87, 0.87]),
        )

        # Initialize with rouge, bertscore, coverage, redundancy (skip faithfulness)
        with patch("src.evaluation.FaithfulnessEvaluator"):
            evaluator = ComprehensiveEvaluator(
                metrics=["rouge", "bertscore", "coverage", "redundancy"]
            )

        with patch("nltk.sent_tokenize", return_value=["Sentence."]):
            results = evaluator.evaluate(
                sample_predictions, sample_references, sample_sources
            )

        # Should have metrics from all evaluators
        assert "rouge1_fmeasure" in results
        assert "bertscore_f1" in results
        assert "coverage_mean" in results
        assert "redundancy_mean" in results

        # Metadata
        assert results["num_samples"] == 3
        assert results["avg_prediction_length"] > 0
        assert results["avg_reference_length"] > 0

    @patch("builtins.print")
    def test_comprehensive_metadata(
        self, mock_print, sample_predictions, sample_references
    ):
        """Test comprehensive evaluator metadata calculation."""
        evaluator = ComprehensiveEvaluator(metrics=["rouge"])
        results = evaluator.evaluate(sample_predictions, sample_references)

        assert results["num_samples"] == len(sample_predictions)

        # Calculate expected average lengths
        expected_pred_len = np.mean([len(p.split()) for p in sample_predictions])
        expected_ref_len = np.mean([len(r.split()) for r in sample_references])

        assert abs(results["avg_prediction_length"] - expected_pred_len) < 0.01
        assert abs(results["avg_reference_length"] - expected_ref_len) < 0.01

    @patch("transformers.pipeline")
    @patch("nltk.sent_tokenize")
    @patch("builtins.print")
    def test_comprehensive_with_faithfulness(
        self,
        mock_print,
        mock_sent_tokenize,
        mock_pipeline,
        sample_predictions,
        sample_sources,
    ):
        """Test comprehensive evaluation including faithfulness."""
        mock_sent_tokenize.return_value = ["Sentence."]
        mock_nli = MagicMock()
        mock_nli.return_value = {"scores": [0.9]}
        mock_pipeline.return_value = mock_nli

        evaluator = ComprehensiveEvaluator(metrics=["faithfulness"])

        with patch("src.evaluation.tqdm", side_effect=lambda x, **kwargs: x):
            results = evaluator.evaluate(
                sample_predictions, sample_predictions, sources=sample_sources
            )

        assert "faithfulness_mean" in results

    @patch("builtins.print")
    def test_comprehensive_without_sources(
        self, mock_print, sample_predictions, sample_references
    ):
        """Test comprehensive evaluation without source documents."""
        evaluator = ComprehensiveEvaluator(metrics=["rouge", "redundancy"])

        with patch("nltk.sent_tokenize", return_value=["Sentence."]):
            results = evaluator.evaluate(sample_predictions, sample_references)

        # Should work without sources for metrics that don't need them
        assert "rouge1_fmeasure" in results
        assert "redundancy_mean" in results
        assert "num_samples" in results
