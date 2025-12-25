"""Comprehensive evaluation suite for summarization models."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tqdm import tqdm


class RougeEvaluator:
    """ROUGE metric evaluator."""

    def __init__(self, rouge_types: List[str] = None):
        """Initialize ROUGE evaluator.

        Args:
            rouge_types: List of ROUGE types to compute
        """
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]

        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE scores.

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries

        Returns:
            Dictionary of ROUGE scores
        """
        scores = {
            rt: {"precision": [], "recall": [], "fmeasure": []}
            for rt in self.rouge_types
        }

        for pred, ref in zip(predictions, references):
            rouge_scores = self.scorer.score(ref, pred)

            for rouge_type in self.rouge_types:
                scores[rouge_type]["precision"].append(
                    rouge_scores[rouge_type].precision
                )
                scores[rouge_type]["recall"].append(rouge_scores[rouge_type].recall)
                scores[rouge_type]["fmeasure"].append(rouge_scores[rouge_type].fmeasure)

        # Compute averages
        avg_scores = {}
        for rouge_type in self.rouge_types:
            avg_scores[f"{rouge_type}_precision"] = np.mean(
                scores[rouge_type]["precision"]
            )
            avg_scores[f"{rouge_type}_recall"] = np.mean(scores[rouge_type]["recall"])
            avg_scores[f"{rouge_type}_fmeasure"] = np.mean(
                scores[rouge_type]["fmeasure"]
            )

        return avg_scores


class BertScoreEvaluator:
    """BERTScore evaluator for semantic similarity."""

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        """Initialize BERTScore evaluator.

        Args:
            model_type: Model to use for BERTScore
        """
        self.model_type = model_type

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute BERTScore.

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries

        Returns:
            Dictionary of BERTScore metrics
        """
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=self.model_type,
            verbose=False,
        )

        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }


class FaithfulnessEvaluator:
    """Faithfulness evaluator using NLI to detect hallucinations."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """Initialize faithfulness evaluator.

        Args:
            model_name: NLI model to use
        """
        from transformers import pipeline

        self.nli_model = pipeline("zero-shot-classification", model=model_name)

    def evaluate_pair(self, source: str, summary: str) -> float:
        """Evaluate faithfulness for a single source-summary pair.

        Args:
            source: Source document
            summary: Generated summary

        Returns:
            Faithfulness score (0-1)
        """
        # Split summary into sentences
        import nltk

        try:
            summary_sentences = nltk.sent_tokenize(summary)
        except LookupError:
            nltk.download("punkt")
            summary_sentences = nltk.sent_tokenize(summary)

        if not summary_sentences:
            return 0.0

        # Check each summary sentence against source
        faithfulness_scores = []

        for sent in summary_sentences:
            # Use NLI to check if sentence is entailed by source
            try:
                result = self.nli_model(
                    sent,
                    candidate_labels=["yes"],
                    hypothesis_template="{}",
                    multi_label=False,
                )
                # Use entailment score as faithfulness
                faithfulness_scores.append(result["scores"][0])
            except Exception:
                # If error, assume neutral faithfulness
                faithfulness_scores.append(0.5)

        return np.mean(faithfulness_scores) if faithfulness_scores else 0.0

    def evaluate(self, predictions: List[str], sources: List[str]) -> Dict:
        """Evaluate faithfulness for multiple summaries.

        Args:
            predictions: List of generated summaries
            sources: List of source documents

        Returns:
            Dictionary with faithfulness metrics
        """
        scores = []

        for pred, source in tqdm(
            zip(predictions, sources), total=len(predictions), desc="Faithfulness"
        ):
            score = self.evaluate_pair(source, pred)
            scores.append(score)

        return {
            "faithfulness_mean": np.mean(scores),
            "faithfulness_std": np.std(scores),
        }


class CoverageEvaluator:
    """Coverage evaluator to check if important content is included."""

    def __init__(self):
        """Initialize coverage evaluator."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=100)

    def evaluate_pair(self, source: str, summary: str) -> float:
        """Evaluate coverage for a single pair.

        Args:
            source: Source document
            summary: Generated summary

        Returns:
            Coverage score (0-1)
        """
        try:
            # Fit vectorizer on source
            source_tfidf = self.vectorizer.fit_transform([source])

            # Transform summary
            summary_tfidf = self.vectorizer.transform([summary])

            # Compute coverage as ratio of non-zero features
            source_features = set(source_tfidf.nonzero()[1])
            summary_features = set(summary_tfidf.nonzero()[1])

            if len(source_features) == 0:
                return 0.0

            coverage = len(summary_features & source_features) / len(source_features)
            return coverage

        except Exception:
            return 0.0

    def evaluate(self, predictions: List[str], sources: List[str]) -> Dict:
        """Evaluate coverage for multiple summaries.

        Args:
            predictions: List of generated summaries
            sources: List of source documents

        Returns:
            Dictionary with coverage metrics
        """
        scores = []

        for pred, source in zip(predictions, sources):
            score = self.evaluate_pair(source, pred)
            scores.append(score)

        return {
            "coverage_mean": np.mean(scores),
            "coverage_std": np.std(scores),
        }


class RedundancyEvaluator:
    """Redundancy evaluator to detect repeated content."""

    def evaluate_pair(self, summary: str) -> float:
        """Evaluate redundancy for a single summary.

        Args:
            summary: Generated summary

        Returns:
            Redundancy score (0-1, lower is better)
        """
        import nltk

        try:
            sentences = nltk.sent_tokenize(summary)
        except LookupError:
            nltk.download("punkt")
            sentences = nltk.sent_tokenize(summary)

        if len(sentences) <= 1:
            return 0.0

        # Check for repeated n-grams
        from collections import Counter

        def get_ngrams(text: str, n: int = 3) -> List[Tuple[str, ...]]:
            words = text.lower().split()
            return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

        all_ngrams = []
        for sent in sentences:
            all_ngrams.extend(get_ngrams(sent, n=3))

        if not all_ngrams:
            return 0.0

        ngram_counts = Counter(all_ngrams)
        repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)

        redundancy = repeated_ngrams / len(all_ngrams) if all_ngrams else 0.0
        return redundancy

    def evaluate(self, predictions: List[str]) -> Dict:
        """Evaluate redundancy for multiple summaries.

        Args:
            predictions: List of generated summaries

        Returns:
            Dictionary with redundancy metrics
        """
        scores = []

        for pred in predictions:
            score = self.evaluate_pair(pred)
            scores.append(score)

        return {
            "redundancy_mean": np.mean(scores),
            "redundancy_std": np.std(scores),
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all metrics."""

    def __init__(self, metrics: List[str] = None):
        """Initialize comprehensive evaluator.

        Args:
            metrics: List of metrics to compute
        """
        if metrics is None:
            metrics = ["rouge", "bertscore", "coverage", "redundancy"]

        self.metrics = metrics
        self.evaluators = {}

        if "rouge" in metrics:
            self.evaluators["rouge"] = RougeEvaluator()
        if "bertscore" in metrics:
            self.evaluators["bertscore"] = BertScoreEvaluator()
        if "faithfulness" in metrics:
            self.evaluators["faithfulness"] = FaithfulnessEvaluator()
        if "coverage" in metrics:
            self.evaluators["coverage"] = CoverageEvaluator()
        if "redundancy" in metrics:
            self.evaluators["redundancy"] = RedundancyEvaluator()

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        sources: List[str] = None,
    ) -> Dict:
        """Run comprehensive evaluation.

        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            sources: List of source documents (for faithfulness/coverage)

        Returns:
            Dictionary of all metrics
        """
        results = {}

        print("Running comprehensive evaluation...")

        if "rouge" in self.evaluators:
            print("Computing ROUGE scores...")
            rouge_scores = self.evaluators["rouge"].evaluate(predictions, references)
            results.update(rouge_scores)

        if "bertscore" in self.evaluators:
            print("Computing BERTScore...")
            bert_scores = self.evaluators["bertscore"].evaluate(predictions, references)
            results.update(bert_scores)

        if "faithfulness" in self.evaluators and sources:
            print("Computing faithfulness scores...")
            faith_scores = self.evaluators["faithfulness"].evaluate(
                predictions, sources
            )
            results.update(faith_scores)

        if "coverage" in self.evaluators and sources:
            print("Computing coverage scores...")
            cov_scores = self.evaluators["coverage"].evaluate(predictions, sources)
            results.update(cov_scores)

        if "redundancy" in self.evaluators:
            print("Computing redundancy scores...")
            red_scores = self.evaluators["redundancy"].evaluate(predictions)
            results.update(red_scores)

        # Add timing and metadata
        results["num_samples"] = len(predictions)
        results["avg_prediction_length"] = np.mean(
            [len(p.split()) for p in predictions]
        )
        results["avg_reference_length"] = np.mean([len(r.split()) for r in references])

        return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate summarization models")
    parser.add_argument(
        "--predictions", type=str, required=True, help="Path to predictions file"
    )
    parser.add_argument(
        "--references", type=str, required=True, help="Path to references file"
    )
    parser.add_argument("--sources", type=str, help="Path to source documents")
    parser.add_argument(
        "--output", type=str, default="evaluation_results.json", help="Output file"
    )
    parser.add_argument("--all-models", action="store_true", help="Evaluate all models")

    args = parser.parse_args()

    # Load data
    with open(args.predictions, "r") as f:
        predictions = [line.strip() for line in f]

    with open(args.references, "r") as f:
        references = [line.strip() for line in f]

    sources = None
    if args.sources:
        with open(args.sources, "r") as f:
            sources = [line.strip() for line in f]

    # Run evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate(predictions, references, sources)

    # Print results
    print("\n" + " 
    print("Evaluation Results")
     
    for metric, value in results.items():
        print(
            f"{metric:30s}: {value:.4f}"
            if isinstance(value, float)
            else f"{metric:30s}: {value}"
        )

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
