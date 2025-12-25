"""Sliding window approach for long document summarization."""

from typing import List, Optional, Tuple

import torch
from transformers import BartForConditionalGeneration, BartTokenizer


class SlidingWindowSummarizer:
    """Sliding window with overlap for long document summarization.

    Processes document in overlapping windows and aggregates results.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        window_size: int = 1024,
        overlap_size: int = 256,
        max_output_length: int = 256,
        aggregation_method: str = "weighted_average",
        device: Optional[str] = None,
    ):
        """Initialize sliding window summarizer.

        Args:
            model_name: Pre-trained model name
            window_size: Size of each window in tokens
            overlap_size: Overlap between windows in tokens
            max_output_length: Maximum output length
            aggregation_method: Method to aggregate window summaries
            device: Device to use
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.window_size = window_size
        self.overlap_size = overlap_size
        self.max_output_length = max_output_length
        self.aggregation_method = aggregation_method

        # Load model and tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def create_windows(self, text: str) -> List[Tuple[str, int, int]]:
        """Create overlapping windows from text.

        Args:
            text: Input document

        Returns:
            List of (window_text, start_pos, end_pos) tuples
        """
        # Tokenize entire document
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        windows = []
        start = 0
        stride = self.window_size - self.overlap_size

        while start < len(tokens):
            end = min(start + self.window_size, len(tokens))
            window_tokens = tokens[start:end]

            # Decode window back to text
            window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)
            windows.append((window_text, start, end))

            if end >= len(tokens):
                break

            start += stride

        return windows

    def summarize_window(self, window_text: str) -> str:
        """Summarize a single window.

        Args:
            window_text: Window text

        Returns:
            Summary of window
        """
        inputs = self.tokenizer(
            window_text,
            max_length=self.window_size,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_output_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def aggregate_weighted_average(self, window_summaries: List[str]) -> str:
        """Aggregate summaries using weighted average approach.

        Args:
            window_summaries: List of window summaries

        Returns:
            Final aggregated summary
        """
        # For simplicity, give more weight to earlier summaries
        # In practice, could use more sophisticated weighting
        if len(window_summaries) == 1:
            return window_summaries[0]

        # Combine summaries with decreasing weights
        combined_text = " ".join(window_summaries)

        # Summarize the combined summaries
        return self.summarize_window(combined_text)

    def aggregate_hierarchical(self, window_summaries: List[str]) -> str:
        """Aggregate summaries hierarchically.

        Args:
            window_summaries: List of window summaries

        Returns:
            Final aggregated summary
        """
        if len(window_summaries) == 1:
            return window_summaries[0]

        # Recursively summarize pairs
        current_summaries = window_summaries[:]

        while len(current_summaries) > 1:
            next_summaries = []

            for i in range(0, len(current_summaries), 2):
                if i + 1 < len(current_summaries):
                    # Combine two summaries
                    combined = current_summaries[i] + " " + current_summaries[i + 1]
                    summary = self.summarize_window(combined)
                    next_summaries.append(summary)
                else:
                    # Odd one out
                    next_summaries.append(current_summaries[i])

            current_summaries = next_summaries

        return current_summaries[0]

    def aggregate_select_top(self, window_summaries: List[str], top_k: int = 3) -> str:
        """Select top-k summaries and combine them.

        Args:
            window_summaries: List of window summaries
            top_k: Number of top summaries to select

        Returns:
            Final aggregated summary
        """
        # For simplicity, select first k summaries
        # In practice, could use importance scoring
        selected = window_summaries[: min(top_k, len(window_summaries))]
        combined = " ".join(selected)

        if len(combined.split()) > self.window_size:
            return self.summarize_window(combined)
        else:
            return combined

    def aggregate(self, window_summaries: List[str]) -> str:
        """Aggregate window summaries based on selected method.

        Args:
            window_summaries: List of window summaries

        Returns:
            Final aggregated summary
        """
        if self.aggregation_method == "weighted_average":
            return self.aggregate_weighted_average(window_summaries)
        elif self.aggregation_method == "hierarchical":
            return self.aggregate_hierarchical(window_summaries)
        elif self.aggregation_method == "select_top":
            return self.aggregate_select_top(window_summaries)
        elif self.aggregation_method == "concat":
            # Simple concatenation
            return " ".join(window_summaries)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def summarize(self, text: str) -> str:
        """Generate summary using sliding window approach.

        Args:
            text: Input document

        Returns:
            Summary text
        """
        # Create windows
        windows = self.create_windows(text)

        if len(windows) == 0:
            return ""

        # Summarize each window
        window_summaries = []
        for window_text, start, end in windows:
            summary = self.summarize_window(window_text)
            window_summaries.append(summary)

        # Aggregate summaries
        final_summary = self.aggregate(window_summaries)

        return final_summary

    def get_window_info(self, text: str) -> dict:
        """Get information about windows created from text.

        Args:
            text: Input document

        Returns:
            Dictionary with window information
        """
        windows = self.create_windows(text)

        total_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))

        info = {
            "total_tokens": total_tokens,
            "num_windows": len(windows),
            "window_size": self.window_size,
            "overlap_size": self.overlap_size,
            "coverage": (
                sum(end - start for _, start, end in windows) / total_tokens
                if total_tokens > 0
                else 0
            ),
        }

        return info


def main():
    """Test sliding window summarizer."""
    test_text = (
        """
    The field of natural language processing has seen tremendous advances in recent years,
    driven largely by the development of transformer-based architectures. These models,
    starting with the original Transformer introduced by Vaswani et al. in 2017, have
    revolutionized how we approach tasks like machine translation, text summarization,
    and question answering. The key innovation was the self-attention mechanism, which
    allows the model to weigh the importance of different words in a sequence when
    processing each word. This was a significant departure from previous recurrent
    architectures like LSTMs and GRUs. Following the original Transformer, we saw the
    development of BERT (Bidirectional Encoder Representations from Transformers) by
    Google, which used a masked language modeling objective to pre-train deep bidirectional
    representations. This was followed by GPT (Generative Pre-trained Transformer) from
    OpenAI, which used a different approach with autoregressive language modeling. These
    models demonstrated that pre-training on large amounts of text data, followed by
    fine-tuning on specific tasks, could achieve state-of-the-art results across a wide
    range of NLP benchmarks.
    """
        * 5
    )  # Repeat to make it longer

    print("Testing Sliding Window Summarizer:")

    try:
        summarizer = SlidingWindowSummarizer(aggregation_method="hierarchical")

        # Get window info
        info = summarizer.get_window_info(test_text)
        print("Window Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Generate summary
        print("\nGenerating summary...")
        summary = summarizer.summarize(test_text)
        print(f"\nSummary:\n{summary}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the transformers library installed.")


if __name__ == "__main__":
    main()
