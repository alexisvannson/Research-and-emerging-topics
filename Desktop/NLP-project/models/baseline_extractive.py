"""Extractive summarization baselines: TextRank and LexRank."""

from typing import List, Optional

import networkx as nx
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextRankSummarizer:
    """TextRank algorithm for extractive summarization.

    Based on PageRank, ranks sentences by their importance in the document.
    """

    def __init__(
        self,
        num_sentences: int = 5,
        similarity_threshold: float = 0.1,
        damping_factor: float = 0.85,
    ):
        """Initialize TextRank summarizer.

        Args:
            num_sentences: Number of sentences to extract
            similarity_threshold: Minimum similarity for edge creation
            damping_factor: Damping factor for PageRank
        """
        self.num_sentences = num_sentences
        self.similarity_threshold = similarity_threshold
        self.damping_factor = damping_factor

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input document

        Returns:
            List of sentences
        """
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt")
            sentences = nltk.sent_tokenize(text)
        return sentences

    def _compute_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Compute sentence similarity matrix using TF-IDF and cosine similarity.

        Args:
            sentences: List of sentences

        Returns:
            Similarity matrix
        """
        # Remove empty sentences
        sentences = [s for s in sentences if s.strip()]

        if len(sentences) < 2:
            return np.zeros((len(sentences), len(sentences)))

        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            # If all sentences are identical or empty
            similarity_matrix = np.zeros((len(sentences), len(sentences)))

        return similarity_matrix

    def _build_graph(self, similarity_matrix: np.ndarray) -> nx.Graph:
        """Build graph from similarity matrix.

        Args:
            similarity_matrix: Sentence similarity matrix

        Returns:
            NetworkX graph
        """
        graph = nx.Graph()
        n = similarity_matrix.shape[0]

        # Add nodes
        for i in range(n):
            graph.add_node(i)

        # Add edges above threshold
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    graph.add_edge(i, j, weight=similarity_matrix[i, j])

        return graph

    def summarize(self, text: str, num_sentences: Optional[int] = None) -> str:
        """Generate extractive summary using TextRank.

        Args:
            text: Input document
            num_sentences: Number of sentences to extract (overrides default)

        Returns:
            Summary text
        """
        num_sentences = num_sentences or self.num_sentences

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= num_sentences:
            return text

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(sentences)

        # Build graph
        graph = self._build_graph(similarity_matrix)

        # Apply PageRank
        try:
            scores = nx.pagerank(graph, alpha=self.damping_factor, max_iter=100)
        except (nx.PowerIterationFailedConvergence, ZeroDivisionError):
            # Fallback: return first N sentences
            return " ".join(sentences[:num_sentences])

        # Rank sentences by score
        ranked_sentences = sorted(
            ((scores.get(i, 0), i, s) for i, s in enumerate(sentences)),
            key=lambda x: x[0],
            reverse=True,
        )

        # Select top sentences and reorder by original position
        selected = sorted(
            [(idx, sent) for _, idx, sent in ranked_sentences[:num_sentences]],
            key=lambda x: x[0],
        )

        summary = " ".join([sent for _, sent in selected])
        return summary


class LexRankSummarizer:
    """LexRank algorithm for extractive summarization.

    Uses continuous sentence importance scores based on eigenvector centrality.
    """

    def __init__(
        self,
        num_sentences: int = 5,
        threshold: float = 0.1,
    ):
        """Initialize LexRank summarizer.

        Args:
            num_sentences: Number of sentences to extract
            threshold: Similarity threshold for edge creation
        """
        self.num_sentences = num_sentences
        self.threshold = threshold

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input document

        Returns:
            List of sentences
        """
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt")
            sentences = nltk.sent_tokenize(text)
        return sentences

    def _compute_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Compute sentence similarity matrix.

        Args:
            sentences: List of sentences

        Returns:
            Similarity matrix
        """
        sentences = [s for s in sentences if s.strip()]

        if len(sentences) < 2:
            return np.zeros((len(sentences), len(sentences)))

        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            similarity_matrix = np.zeros((len(sentences), len(sentences)))

        return similarity_matrix

    def _create_markov_matrix(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Create Markov matrix from similarity matrix.

        Args:
            similarity_matrix: Sentence similarity matrix

        Returns:
            Markov matrix (row-stochastic)
        """
        n = similarity_matrix.shape[0]
        markov_matrix = np.zeros((n, n))

        for i in range(n):
            # Find sentences above threshold
            connected = similarity_matrix[i] > self.threshold
            num_connected = np.sum(connected)

            if num_connected > 0:
                # Distribute probability uniformly among connected sentences
                markov_matrix[i, connected] = 1.0 / num_connected
            else:
                # If no connections, uniform distribution
                markov_matrix[i, :] = 1.0 / n

        return markov_matrix

    def _power_method(
        self, matrix: np.ndarray, max_iter: int = 100, tol: float = 1e-6
    ) -> np.ndarray:
        """Compute principal eigenvector using power method.

        Args:
            matrix: Input matrix
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Principal eigenvector
        """
        n = matrix.shape[0]
        vector = np.ones(n) / n

        for _ in range(max_iter):
            new_vector = matrix.T @ vector
            # Normalize
            new_vector = new_vector / np.linalg.norm(new_vector, 1)

            if np.linalg.norm(new_vector - vector, 1) < tol:
                break

            vector = new_vector

        return vector

    def summarize(self, text: str, num_sentences: Optional[int] = None) -> str:
        """Generate extractive summary using LexRank.

        Args:
            text: Input document
            num_sentences: Number of sentences to extract

        Returns:
            Summary text
        """
        num_sentences = num_sentences or self.num_sentences

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= num_sentences:
            return text

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(sentences)

        # Create Markov matrix
        markov_matrix = self._create_markov_matrix(similarity_matrix)

        # Compute sentence importance scores
        scores = self._power_method(markov_matrix)

        # Rank sentences
        ranked_sentences = sorted(
            enumerate(sentences), key=lambda x: scores[x[0]], reverse=True
        )

        # Select top sentences and reorder by original position
        selected = sorted(
            [(idx, sent) for idx, sent in ranked_sentences[:num_sentences]],
            key=lambda x: x[0],
        )

        summary = " ".join([sent for _, sent in selected])
        return summary


def main():
    """Test extractive summarizers."""
    test_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence concerned with the interactions between computers and
    human language. In particular, how to program computers to process and analyze
    large amounts of natural language data. The goal is a computer capable of
    understanding the contents of documents, including the contextual nuances of
    the language within them. The technology can then accurately extract information
    and insights contained in the documents as well as categorize and organize the
    documents themselves. Challenges in natural language processing frequently involve
    speech recognition, natural language understanding, and natural language generation.
    Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing
    published an article titled "Computing Machinery and Intelligence" which proposed
    what is now called the Turing test as a criterion of intelligence. The Georgetown
    experiment in 1954 involved fully automatic translation of more than sixty Russian
    sentences into English. The authors claimed that within three or five years,
    machine translation would be a solved problem. However, real progress was much slower,
    and after the ALPAC report in 1966, which found that ten years of research had failed
    to fulfill the expectations, funding for machine translation was dramatically reduced.
    """

    print("Testing TextRank Summarizer:")

    textrank = TextRankSummarizer(num_sentences=3)
    summary = textrank.summarize(test_text)
    print(summary)

    print("\n\nTesting LexRank Summarizer:")

    lexrank = LexRankSummarizer(num_sentences=3)
    summary = lexrank.summarize(test_text)
    print(summary)


if __name__ == "__main__":
    main()
