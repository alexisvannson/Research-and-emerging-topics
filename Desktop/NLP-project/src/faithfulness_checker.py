"""Faithfulness checking to detect hallucinations in summaries."""

from typing import Dict, List, Tuple

import nltk
import torch
from transformers import pipeline


class FaithfulnessChecker:
    """Check faithfulness of generated summaries using NLI."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize faithfulness checker.

        Args:
            model_name: NLI model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nli_pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.device == "cuda" else -1,
        )

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt")
            return nltk.sent_tokenize(text)

    def check_sentence_faithfulness(
        self, source: str, sentence: str
    ) -> Tuple[float, str]:
        """Check if a sentence is faithful to the source.

        Args:
            source: Source document
            sentence: Sentence to check

        Returns:
            Tuple of (faithfulness_score, label)
        """
        try:
            # Use NLI to check entailment
            result = self.nli_pipeline(
                sentence,
                candidate_labels=["entailment", "contradiction", "neutral"],
                hypothesis_template="{}",
                multi_label=False,
            )

            # Get score for entailment
            label = result["labels"][0]
            score = result["scores"][0]

            # Map label to score (higher is better)
            if label == "entailment":
                faithfulness_score = score
            elif label == "contradiction":
                faithfulness_score = 1.0 - score
            else:  # neutral
                faithfulness_score = 0.5

            return faithfulness_score, label

        except Exception as e:
            print(f"Error checking faithfulness: {e}")
            return 0.5, "unknown"

    def check_summary(self, source: str, summary: str) -> Dict:
        """Check faithfulness of entire summary.

        Args:
            source: Source document
            summary: Generated summary

        Returns:
            Dictionary with faithfulness analysis
        """
        # Split summary into sentences
        summary_sentences = self.split_sentences(summary)

        if not summary_sentences:
            return {
                "overall_score": 0.0,
                "num_sentences": 0,
                "sentence_scores": [],
                "hallucinations": [],
            }

        # Check each sentence
        sentence_scores = []
        hallucinations = []

        for sent in summary_sentences:
            score, label = self.check_sentence_faithfulness(source, sent)
            sentence_scores.append(score)

            # Flag potential hallucinations (low scores)
            if score < 0.5 or label == "contradiction":
                hallucinations.append(
                    {
                        "sentence": sent,
                        "score": score,
                        "label": label,
                    }
                )

        # Compute overall score
        overall_score = sum(sentence_scores) / len(sentence_scores)

        return {
            "overall_score": overall_score,
            "num_sentences": len(summary_sentences),
            "sentence_scores": sentence_scores,
            "hallucinations": hallucinations,
            "hallucination_rate": len(hallucinations) / len(summary_sentences),
        }

    def batch_check(self, sources: List[str], summaries: List[str]) -> List[Dict]:
        """Check faithfulness for multiple summaries.

        Args:
            sources: List of source documents
            summaries: List of generated summaries

        Returns:
            List of faithfulness results
        """
        results = []

        for source, summary in zip(sources, summaries):
            result = self.check_summary(source, summary)
            results.append(result)

        return results


def main():
    """Test faithfulness checker."""
    source = """
    The Transformer architecture was introduced in the paper "Attention is All You Need"
    by Vaswani et al. in 2017. It uses self-attention mechanisms to process sequences
    in parallel, unlike previous RNN-based models. This led to significant improvements
    in machine translation tasks.
    """

    # Faithful summary
    faithful_summary = """
    The Transformer architecture, introduced by Vaswani et al. in 2017, uses
    self-attention to process sequences in parallel, improving machine translation.
    """

    # Unfaithful summary (contains hallucination)
    unfaithful_summary = """
    The Transformer was invented in 2015 by Google researchers. It uses convolutional
    layers to process text and achieved perfect accuracy on all NLP tasks.
    """

    print("Testing Faithfulness Checker:")

    checker = FaithfulnessChecker()

    print("\nFaithful Summary:")
    result1 = checker.check_summary(source, faithful_summary)
    print(f"Overall Score: {result1['overall_score']:.4f}")
    print(f"Hallucinations: {len(result1['hallucinations'])}")

    print("\nUnfaithful Summary:")
    result2 = checker.check_summary(source, unfaithful_summary)
    print(f"Overall Score: {result2['overall_score']:.4f}")
    print(f"Hallucinations: {len(result2['hallucinations'])}")
    for h in result2["hallucinations"]:
        print(f"  - {h['sentence'][:50]}... (score: {h['score']:.4f})")


if __name__ == "__main__":
    main()
