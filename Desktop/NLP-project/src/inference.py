"""Inference utilities for summarization models."""

import time
from typing import Any, Dict, List, Optional

from models.baseline_abstractive import BARTChunkSummarizer
from models.baseline_extractive import LexRankSummarizer, TextRankSummarizer
from models.hierarchical_transformer import HierarchicalTransformerSummarizer
from models.longformer_summarizer import LongformerSummarizer
from models.sliding_window import SlidingWindowSummarizer


class SummarizationInference:
    """Unified inference interface for all summarization models."""

    def __init__(self, model_type: str, config: Optional[Dict] = None):
        """Initialize inference engine.

        Args:
            model_type: Type of model (textrank, lexrank, bart, hierarchical, longformer, sliding)
            config: Model configuration
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = self._load_model()

    def _load_model(self):
        """Load the specified model."""
        if self.model_type == "textrank":
            return TextRankSummarizer(**self.config.get("extractive", {}))

        elif self.model_type == "lexrank":
            return LexRankSummarizer(**self.config.get("extractive", {}))

        elif self.model_type == "bart":
            return BARTChunkSummarizer(**self.config.get("abstractive", {}))

        elif self.model_type == "hierarchical":
            return HierarchicalTransformerSummarizer(
                **self.config.get("hierarchical", {})
            )

        elif self.model_type == "longformer":
            return LongformerSummarizer(**self.config.get("longformer", {}))

        elif self.model_type == "sliding":
            return SlidingWindowSummarizer(**self.config.get("sliding", {}))

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def summarize(self, text: str, return_metrics: bool = False) -> Dict[str, Any]:
        """Generate summary with optional metrics.

        Args:
            text: Input document
            return_metrics: Whether to return performance metrics

        Returns:
            Dictionary with summary and optional metrics
        """
        start_time = time.time()

        # Generate summary
        summary = self.model.summarize(text)

        inference_time = time.time() - start_time

        result = {
            "summary": summary,
        }

        if return_metrics:
            # Calculate metrics
            source_tokens = len(text.split())
            summary_tokens = len(summary.split())

            result["metrics"] = {
                "inference_time": inference_time,
                "source_tokens": source_tokens,
                "summary_tokens": summary_tokens,
                "compression_ratio": (
                    source_tokens / summary_tokens if summary_tokens > 0 else 0
                ),
            }

        return result

    def batch_summarize(
        self, texts: List[str], show_progress: bool = True
    ) -> List[Dict]:
        """Summarize multiple documents.

        Args:
            texts: List of input documents
            show_progress: Whether to show progress bar

        Returns:
            List of results
        """
        results = []

        if show_progress:
            from tqdm import tqdm

            texts = tqdm(texts, desc="Summarizing")

        for text in texts:
            result = self.summarize(text, return_metrics=True)
            results.append(result)

        return results


def load_model_from_checkpoint(checkpoint_path: str, model_type: str, config: Dict):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model_type: Model type
        config: Model configuration

    Returns:
        Loaded model
    """
    # This would load a trained model checkpoint
    # For now, just create a new model
    inference = SummarizationInference(model_type, config)
    return inference


def main():
    """Test inference."""
    test_text = """
    The field of natural language processing has seen tremendous advances in recent years.
    These advances have been driven largely by the development of transformer-based
    architectures. The original Transformer, introduced by Vaswani et al. in 2017,
    revolutionized how we approach sequence-to-sequence tasks.
    """

    # Test TextRank
    print("Testing TextRank Inference:")

    inference = SummarizationInference("textrank", {"extractive": {"num_sentences": 2}})
    result = inference.summarize(test_text, return_metrics=True)
    print(f"Summary: {result['summary']}")
    print(f"Metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
