"""Longformer-based summarizer for long documents.

Uses LED (Longformer Encoder-Decoder) which can handle up to 16K tokens
with sparse attention patterns.
"""

from typing import List, Optional

import torch
from transformers import LEDForConditionalGeneration, LEDTokenizer


class LongformerSummarizer:
    """LED-based summarizer for long documents with sparse attention."""

    def __init__(
        self,
        model_name: str = "allenai/led-large-16384",
        max_input_length: int = 16384,
        max_output_length: int = 1024,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        device: Optional[str] = None,
    ):
        """Initialize Longformer summarizer.

        Args:
            model_name: Pre-trained LED model name
            max_input_length: Maximum input length (up to 16384)
            max_output_length: Maximum output length
            num_beams: Number of beams for generation
            length_penalty: Length penalty for generation
            no_repeat_ngram_size: No repeat n-gram size
            device: Device to use
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = LEDTokenizer.from_pretrained(model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def set_global_attention(
        self, input_ids: torch.Tensor, mode: str = "first_sentence"
    ) -> torch.Tensor:
        """Set global attention mask for the input.

        Args:
            input_ids: Input token IDs
            mode: Global attention mode (first_sentence, uniform, keywords)

        Returns:
            Global attention mask
        """
        batch_size, seq_len = input_ids.shape
        global_attention_mask = torch.zeros_like(input_ids)

        if mode == "first_sentence":
            # Set global attention on first sentence tokens (rough approximation)
            # Typically first 64 tokens
            num_global_tokens = min(64, seq_len)
            global_attention_mask[:, :num_global_tokens] = 1

        elif mode == "uniform":
            # Set global attention uniformly across the document
            step = max(1, seq_len // 64)
            global_attention_mask[:, ::step] = 1

        elif mode == "keywords":
            # This would require keyword extraction, simplified version
            # Set attention on start, middle, and end
            global_attention_mask[:, 0] = 1  # Start
            if seq_len > 100:
                global_attention_mask[:, seq_len // 2] = 1  # Middle
                global_attention_mask[:, -1] = 1  # End

        return global_attention_mask

    def summarize(
        self,
        text: str,
        global_attention_mode: str = "first_sentence",
    ) -> str:
        """Generate summary for long document.

        Args:
            text: Input document
            global_attention_mode: Mode for global attention

        Returns:
            Summary text
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Set global attention mask
        global_attention_mask = self.set_global_attention(
            input_ids, mode=global_attention_mode
        )
        global_attention_mask = global_attention_mask.to(self.device)

        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                max_length=self.max_output_length,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                early_stopping=True,
            )

        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def batch_summarize(
        self,
        texts: List[str],
        batch_size: int = 1,
        global_attention_mode: str = "first_sentence",
    ) -> List[str]:
        """Summarize multiple documents in batches.

        Args:
            texts: List of input documents
            batch_size: Batch size for processing
            global_attention_mode: Mode for global attention

        Returns:
            List of summaries
        """
        summaries = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            for text in batch_texts:
                summary = self.summarize(text, global_attention_mode)
                summaries.append(summary)

        return summaries

    def analyze_attention(
        self,
        text: str,
        global_attention_mode: str = "first_sentence",
    ) -> dict:
        """Analyze attention patterns for a document.

        Args:
            text: Input document
            global_attention_mode: Mode for global attention

        Returns:
            Dictionary with attention analysis
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Set global attention
        global_attention_mask = self.set_global_attention(
            input_ids, mode=global_attention_mode
        )
        global_attention_mask = global_attention_mask.to(self.device)

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                output_attentions=True,
            )

        # Collect attention analysis
        analysis = {
            "num_tokens": input_ids.shape[1],
            "num_global_tokens": global_attention_mask.sum().item(),
            "global_attention_ratio": global_attention_mask.sum().item()
            / input_ids.shape[1],
            "attention_layers": len(outputs.attentions) if outputs.attentions else 0,
        }

        return analysis


class BigBirdSummarizer:
    """BigBird-based summarizer (alternative to Longformer)."""

    def __init__(
        self,
        model_name: str = "google/bigbird-pegasus-large-arxiv",
        max_input_length: int = 4096,
        max_output_length: int = 1024,
        device: Optional[str] = None,
    ):
        """Initialize BigBird summarizer.

        Args:
            model_name: Pre-trained BigBird model name
            max_input_length: Maximum input length
            max_output_length: Maximum output length
            device: Device to use
        """
        from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def summarize(self, text: str) -> str:
        """Generate summary using BigBird.

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
                **inputs,
                max_length=self.max_output_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


def main():
    """Test Longformer summarizer."""
    test_text = (
        """
    The field of natural language processing has seen tremendous advances in recent years,
    driven largely by the development of transformer-based architectures. These models,
    starting with the original Transformer introduced by Vaswani et al. in 2017, have
    revolutionized how we approach tasks like machine translation, text summarization,
    and question answering. The key innovation was the self-attention mechanism, which
    allows the model to weigh the importance of different words in a sequence when
    processing each word. This was a significant departure from previous recurrent
    architectures like LSTMs and GRUs.
    """
        * 10
    )  # Repeat to make it longer

    print("Testing Longformer Summarizer:")

    print(f"Input length: {len(test_text.split())} words")

    try:
        summarizer = LongformerSummarizer()
        summary = summarizer.summarize(test_text)
        print(f"\nSummary length: {len(summary.split())} words")
        print(f"Summary:\n{summary}")

        # Analyze attention
        print("Attention Analysis:")
        analysis = summarizer.analyze_attention(test_text)
        for key, value in analysis.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        print("This model requires significant GPU memory to run.")


if __name__ == "__main__":
    main()
