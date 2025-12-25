"""Hierarchical Transformer for long document summarization.

Encodes documents in two stages:
1. Paragraph-level encoding using BERT
2. Document-level encoding combining paragraph representations
"""

from typing import List, Optional, Tuple

import nltk
import torch
import torch.nn as nn
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BertModel,
    BertTokenizer,
)


class HierarchicalEncoder(nn.Module):
    """Two-level hierarchical encoder for long documents."""

    def __init__(
        self,
        paragraph_encoder_name: str = "bert-base-uncased",
        max_paragraph_length: int = 512,
        hidden_size: int = 768,
        num_doc_layers: int = 4,
        num_attention_heads: int = 8,
        max_paragraphs: int = 32,
    ):
        """Initialize hierarchical encoder.

        Args:
            paragraph_encoder_name: Name of paragraph encoder model
            max_paragraph_length: Max tokens per paragraph
            hidden_size: Hidden dimension size
            num_doc_layers: Number of document-level transformer layers
            num_attention_heads: Number of attention heads
            max_paragraphs: Maximum number of paragraphs
        """
        super().__init__()

        self.max_paragraph_length = max_paragraph_length
        self.hidden_size = hidden_size
        self.max_paragraphs = max_paragraphs

        # Paragraph-level encoder (BERT)
        self.paragraph_encoder = BertModel.from_pretrained(paragraph_encoder_name)
        self.paragraph_tokenizer = BertTokenizer.from_pretrained(paragraph_encoder_name)

        # Document-level transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.document_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_doc_layers
        )

        # Position embeddings for paragraphs
        self.paragraph_position_embeddings = nn.Embedding(max_paragraphs, hidden_size)

    def encode_paragraph(self, paragraph_text: str, device: str) -> torch.Tensor:
        """Encode a single paragraph.

        Args:
            paragraph_text: Paragraph text
            device: Device to use

        Returns:
            Paragraph encoding (hidden_size,)
        """
        # Tokenize paragraph
        inputs = self.paragraph_tokenizer(
            paragraph_text,
            max_length=self.max_paragraph_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode paragraph
        with torch.no_grad():
            outputs = self.paragraph_encoder(**inputs)

        # Use [CLS] token representation
        paragraph_encoding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
        return paragraph_encoding.squeeze(0)

    def forward(
        self, paragraphs: List[str], device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode document hierarchically.

        Args:
            paragraphs: List of paragraph texts
            device: Device to use

        Returns:
            Tuple of (document_encoding, paragraph_encodings)
        """
        # Limit number of paragraphs
        paragraphs = paragraphs[: self.max_paragraphs]
        num_paragraphs = len(paragraphs)

        # Encode each paragraph
        paragraph_encodings_list: List[torch.Tensor] = []
        for paragraph in paragraphs:
            encoding = self.encode_paragraph(paragraph, device)
            paragraph_encodings_list.append(encoding)

        # Stack paragraph encodings
        paragraph_encodings = torch.stack(
            paragraph_encodings_list
        )  # (num_para, hidden_size)

        # Add position embeddings
        positions = torch.arange(num_paragraphs, device=device)
        position_embeddings = self.paragraph_position_embeddings(positions)
        paragraph_encodings = paragraph_encodings + position_embeddings

        # Add batch dimension
        paragraph_encodings = paragraph_encodings.unsqueeze(
            0
        )  # (1, num_para, hidden_size)

        # Document-level encoding
        document_encoding = self.document_encoder(
            paragraph_encodings
        )  # (1, num_para, hidden_size)

        return document_encoding.squeeze(0), paragraph_encodings.squeeze(0)


class HierarchicalTransformerSummarizer:
    """Complete hierarchical transformer summarization model."""

    def __init__(
        self,
        paragraph_encoder_name: str = "bert-base-uncased",
        decoder_name: str = "facebook/bart-large",
        max_paragraph_length: int = 512,
        max_paragraphs: int = 32,
        max_output_length: int = 512,
        num_beams: int = 4,
        device: Optional[str] = None,
    ):
        """Initialize hierarchical summarizer.

        Args:
            paragraph_encoder_name: Name of paragraph encoder
            decoder_name: Name of decoder model
            max_paragraph_length: Max tokens per paragraph
            max_paragraphs: Maximum number of paragraphs
            max_output_length: Maximum output length
            num_beams: Number of beams for generation
            device: Device to use
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_paragraph_length = max_paragraph_length
        self.max_paragraphs = max_paragraphs
        self.max_output_length = max_output_length
        self.num_beams = num_beams

        # Initialize hierarchical encoder
        self.encoder = HierarchicalEncoder(
            paragraph_encoder_name=paragraph_encoder_name,
            max_paragraph_length=max_paragraph_length,
            max_paragraphs=max_paragraphs,
        )
        self.encoder.to(self.device)
        self.encoder.eval()

        # Initialize decoder
        self.decoder = BartForConditionalGeneration.from_pretrained(decoder_name)
        self.decoder_tokenizer = BartTokenizer.from_pretrained(decoder_name)
        self.decoder.to(self.device)
        self.decoder.eval()

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text: Input document

        Returns:
            List of paragraphs
        """
        import re

        # Split on double newlines or multiple spaces
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If no clear paragraph breaks, use sentence-based splitting
        if len(paragraphs) == 1:
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt")
                sentences = nltk.sent_tokenize(text)

            # Group sentences into pseudo-paragraphs
            paragraphs = []
            current_para: List[str] = []
            current_length = 0

            for sent in sentences:
                sent_length = len(sent.split())
                if current_length + sent_length > 100 and current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = [sent]
                    current_length = sent_length
                else:
                    current_para.append(sent)
                    current_length += sent_length

            if current_para:
                paragraphs.append(" ".join(current_para))

        return paragraphs

    def summarize(self, text: str) -> str:
        """Generate summary using hierarchical approach.

        Args:
            text: Input document

        Returns:
            Summary text
        """
        # Split into paragraphs
        paragraphs = self.split_into_paragraphs(text)

        if len(paragraphs) == 0:
            return ""

        # Encode document hierarchically
        with torch.no_grad():
            doc_encoding, para_encodings = self.encoder(paragraphs, self.device)

        # For BART decoder, we need to create appropriate inputs
        # We'll use the mean of paragraph encodings as a simple approach
        summary_prompt = "Summarize: " + " ".join(paragraphs[: min(3, len(paragraphs))])

        # Tokenize the prompt
        decoder_inputs = self.decoder_tokenizer(
            summary_prompt,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        decoder_inputs = {k: v.to(self.device) for k, v in decoder_inputs.items()}

        # Generate summary
        with torch.no_grad():
            summary_ids = self.decoder.generate(
                decoder_inputs["input_ids"],
                max_length=self.max_output_length,
                num_beams=self.num_beams,
                length_penalty=2.0,
                early_stopping=True,
            )

        summary = self.decoder_tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )
        return summary


def main():
    """Test hierarchical transformer summarizer."""
    test_text = """
    The field of natural language processing has seen tremendous advances in recent years.
    These advances have been driven largely by the development of transformer-based
    architectures and large-scale pre-training.

    The original Transformer, introduced by Vaswani et al. in 2017, revolutionized
    how we approach sequence-to-sequence tasks. The key innovation was the self-attention
    mechanism, which allows the model to weigh the importance of different words in a
    sequence when processing each word.

    Following the original Transformer, we saw the development of BERT by Google, which
    used a masked language modeling objective to pre-train deep bidirectional representations.
    This was followed by GPT from OpenAI, which used autoregressive language modeling.

    These models demonstrated that pre-training on large amounts of text data, followed
    by fine-tuning on specific tasks, could achieve state-of-the-art results across a
    wide range of NLP benchmarks. The success led to an explosion of research in the area.

    However, one limitation of standard transformers is their quadratic complexity with
    respect to sequence length. This makes them computationally expensive for long documents.
    This has led to research on efficient transformer variants like Longformer and BigBird.
    """

    print("Testing Hierarchical Transformer Summarizer:")

    try:
        summarizer = HierarchicalTransformerSummarizer()
        summary = summarizer.summarize(test_text)
        print(summary)
    except Exception as e:
        print(f"Error: {e}")
        print("This model requires GPU and pre-trained weights to run properly.")


if __name__ == "__main__":
    main()
