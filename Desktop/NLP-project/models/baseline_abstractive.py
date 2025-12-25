"""Abstractive summarization baseline using BART with chunking."""

from typing import List, Optional

import torch
from transformers import BartForConditionalGeneration, BartTokenizer


class BARTChunkSummarizer:
    """BART-based abstractive summarizer for long documents.

    Handles long documents by chunking and aggregating summaries.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        max_input_length: int = 1024,
        max_output_length: int = 256,
        min_output_length: int = 50,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        aggregation_method: str = "concat",
        device: Optional[str] = None,
    ):
        """Initialize BART summarizer.

        Args:
            model_name: Pre-trained BART model name
            max_input_length: Maximum input tokens
            max_output_length: Maximum output tokens
            min_output_length: Minimum output tokens
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks
            aggregation_method: Method to aggregate chunk summaries
            device: Device to use (cuda/cpu)
        """
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.aggregation_method = aggregation_method

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode chunk back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Move start position
            if end >= len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _summarize_chunk(self, chunk: str) -> str:
        """Summarize a single chunk.

        Args:
            chunk: Text chunk

        Returns:
            Summary of chunk
        """
        # Tokenize
        inputs = self.tokenizer(
            chunk,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_output_length,
                min_length=self.min_output_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def _aggregate_summaries(self, summaries: List[str]) -> str:
        """Aggregate chunk summaries into final summary.

        Args:
            summaries: List of chunk summaries

        Returns:
            Final aggregated summary
        """
        if self.aggregation_method == "concat":
            # Simply concatenate summaries
            return " ".join(summaries)

        elif self.aggregation_method == "summary_of_summaries":
            # Generate summary of the summaries
            combined = " ".join(summaries)
            return self._summarize_chunk(combined)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def summarize(self, text: str) -> str:
        """Generate abstractive summary of long document.

        Args:
            text: Input document

        Returns:
            Summary text
        """
        # Check if text fits in single chunk
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_input_length:
            # Can summarize directly
            return self._summarize_chunk(text)

        # Split into chunks
        chunks = self._chunk_text(text)

        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self._summarize_chunk(chunk)
            chunk_summaries.append(summary)

        # Aggregate summaries
        final_summary = self._aggregate_summaries(chunk_summaries)

        return final_summary

    def batch_summarize(self, texts: List[str]) -> List[str]:
        """Summarize multiple documents.

        Args:
            texts: List of input documents

        Returns:
            List of summaries
        """
        summaries = []
        for text in texts:
            summary = self.summarize(text)
            summaries.append(summary)
        return summaries


class PEGASUSSummarizer:
    """PEGASUS-based abstractive summarizer.

    Alternative to BART, specifically designed for summarization.
    """

    def __init__(
        self,
        model_name: str = "google/pegasus-large",
        max_input_length: int = 1024,
        max_output_length: int = 256,
        device: Optional[str] = None,
    ):
        """Initialize PEGASUS summarizer.

        Args:
            model_name: Pre-trained PEGASUS model name
            max_input_length: Maximum input tokens
            max_output_length: Maximum output tokens
            device: Device to use (cuda/cpu)
        """
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def summarize(self, text: str) -> str:
        """Generate abstractive summary.

        Args:
            text: Input document

        Returns:
            Summary text
        """
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
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


def main():
    """Test abstractive summarizers."""
    test_text = """
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
    range of NLP benchmarks. The success of these models led to an explosion of research
    in the area, with new models and techniques being published at a rapid pace. However,
    one limitation of standard transformers is their quadratic complexity with respect to
    sequence length, which makes them computationally expensive for long documents. This
    has led to research on efficient transformer variants like Longformer and BigBird.
    """

    print("Testing BART Chunk Summarizer:")

    try:
        bart = BARTChunkSummarizer()
        summary = bart.summarize(test_text)
        print(summary)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the transformers library installed.")


if __name__ == "__main__":
    main()
