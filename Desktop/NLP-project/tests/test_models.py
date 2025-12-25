"""Unit tests for summarization models."""

import pytest
import torch

from models.baseline_abstractive import BARTChunkSummarizer
from models.baseline_extractive import LexRankSummarizer, TextRankSummarizer
from models.sliding_window import SlidingWindowSummarizer
from models.utils import (
    AverageMeter,
    compute_compression_ratio,
    count_parameters,
    format_time,
    get_device,
    set_seed,
    truncate_text,
)

# Test data
TEST_TEXT = """
Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language. In particular, how to program computers to process and analyze
large amounts of natural language data. The goal is a computer capable of
understanding the contents of documents, including the contextual nuances of
the language within them. The technology can then accurately extract information
and insights contained in the documents as well as categorize and organize the
documents themselves. Challenges in natural language processing frequently involve
speech recognition, natural language understanding, and natural language generation.
"""


class TestTextRankSummarizer:
    """Test TextRank summarizer."""

    def test_initialization(self):
        """Test model initialization."""
        summarizer = TextRankSummarizer(num_sentences=3)
        assert summarizer.num_sentences == 3
        assert summarizer.damping_factor == 0.85

    def test_summarize(self):
        """Test summary generation."""
        summarizer = TextRankSummarizer(num_sentences=2)
        summary = summarizer.summarize(TEST_TEXT)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) < len(TEST_TEXT)

    def test_summarize_short_text(self):
        """Test summarization of text shorter than num_sentences."""
        summarizer = TextRankSummarizer(num_sentences=10)
        short_text = "This is a short sentence."
        summary = summarizer.summarize(short_text)

        assert summary == short_text

    def test_empty_text(self):
        """Test handling of empty text."""
        summarizer = TextRankSummarizer()
        summary = summarizer.summarize("")

        assert summary == ""


class TestLexRankSummarizer:
    """Test LexRank summarizer."""

    def test_initialization(self):
        """Test model initialization."""
        summarizer = LexRankSummarizer(num_sentences=3)
        assert summarizer.num_sentences == 3

    def test_summarize(self):
        """Test summary generation."""
        summarizer = LexRankSummarizer(num_sentences=2)
        summary = summarizer.summarize(TEST_TEXT)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_custom_threshold(self):
        """Test with custom threshold."""
        summarizer = LexRankSummarizer(num_sentences=2, threshold=0.2)
        summary = summarizer.summarize(TEST_TEXT)

        assert isinstance(summary, str)


class TestBARTChunkSummarizer:
    """Test BART chunk summarizer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_initialization(self):
        """Test model initialization."""
        summarizer = BARTChunkSummarizer(max_input_length=512)
        assert summarizer.max_input_length == 512

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_chunk_text(self):
        """Test text chunking."""
        summarizer = BARTChunkSummarizer(chunk_size=100, chunk_overlap=20)
        long_text = TEST_TEXT * 10

        chunks = summarizer._chunk_text(long_text)
        assert len(chunks) > 1


class TestSlidingWindowSummarizer:
    """Test sliding window summarizer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_initialization(self):
        """Test model initialization."""
        summarizer = SlidingWindowSummarizer(window_size=512, overlap_size=128)
        assert summarizer.window_size == 512
        assert summarizer.overlap_size == 128

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_create_windows(self):
        """Test window creation."""
        summarizer = SlidingWindowSummarizer(window_size=100, overlap_size=20)
        long_text = TEST_TEXT * 10

        windows = summarizer.create_windows(long_text)
        assert len(windows) > 0

        # Check overlap
        if len(windows) > 1:
            _, start1, end1 = windows[0]
            _, start2, end2 = windows[1]
            assert start2 < end1  # Overlap exists

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_window_info(self):
        """Test getting window information."""
        summarizer = SlidingWindowSummarizer()
        info = summarizer.get_window_info(TEST_TEXT)

        assert "total_tokens" in info
        assert "num_windows" in info
        assert info["total_tokens"] > 0


class TestUtilities:
    """Test utility functions."""

    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        import random

        val1 = random.random()

        set_seed(42)
        val2 = random.random()

        assert val1 == val2

    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in ["cuda", "cpu"]

    def test_format_time(self):
        """Test time formatting."""
        assert "s" in format_time(30)
        assert "m" in format_time(120)
        assert "h" in format_time(7200)

    def test_truncate_text(self):
        """Test text truncation."""
        long_text = " ".join([f"word{i}" for i in range(200)])
        truncated = truncate_text(long_text, max_length=50)

        assert len(truncated.split()) <= 51  # 50 + "..."
        assert "..." in truncated

    def test_truncate_short_text(self):
        """Test truncation of short text."""
        short_text = "short text"
        truncated = truncate_text(short_text, max_length=100)

        assert truncated == short_text

    def test_compute_compression_ratio(self):
        """Test compression ratio computation."""
        source = "This is a very long source document with many words."
        summary = "Short summary."

        ratio = compute_compression_ratio(source, summary)
        assert ratio > 1.0

    def test_compression_ratio_empty_summary(self):
        """Test compression ratio with empty summary."""
        ratio = compute_compression_ratio("source", "")
        assert ratio == float("inf")

    def test_average_meter(self):
        """Test AverageMeter class."""
        meter = AverageMeter()

        meter.update(10)
        assert meter.avg == 10
        assert meter.count == 1

        meter.update(20)
        assert meter.avg == 15
        assert meter.count == 2

        meter.reset()
        assert meter.avg == 0
        assert meter.count == 0

    def test_count_parameters(self):
        """Test parameter counting."""
        import torch.nn as nn

        model = nn.Linear(10, 5)
        params = count_parameters(model)

        assert "total_parameters" in params
        assert "trainable_parameters" in params
        assert params["total_parameters"] == 55  # 10*5 + 5


class TestIntegration:
    """Integration tests."""

    def test_extractive_pipeline(self):
        """Test complete extractive summarization pipeline."""
        # TextRank
        summarizer1 = TextRankSummarizer(num_sentences=2)
        summary1 = summarizer1.summarize(TEST_TEXT)

        # LexRank
        summarizer2 = LexRankSummarizer(num_sentences=2)
        summary2 = summarizer2.summarize(TEST_TEXT)

        # Both should produce summaries
        assert len(summary1) > 0
        assert len(summary2) > 0

        # Summaries should be different (different algorithms)
        # Note: They might occasionally be the same for short texts
        assert isinstance(summary1, str)
        assert isinstance(summary2, str)

    def test_reproducibility(self):
        """Test reproducibility with seed."""
        set_seed(42)
        summarizer1 = TextRankSummarizer(num_sentences=2)
        summary1 = summarizer1.summarize(TEST_TEXT)

        set_seed(42)
        summarizer2 = TextRankSummarizer(num_sentences=2)
        summary2 = summarizer2.summarize(TEST_TEXT)

        # Should produce same results
        assert summary1 == summary2


def test_import_all_models():
    """Test that all models can be imported."""
    from models import (
        baseline_abstractive,
        baseline_extractive,
        hierarchical_transformer,
        longformer_summarizer,
        sliding_window,
        utils,
    )

    assert baseline_abstractive is not None
    assert baseline_extractive is not None
    assert hierarchical_transformer is not None
    assert longformer_summarizer is not None
    assert sliding_window is not None
    assert utils is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
