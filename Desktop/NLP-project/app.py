"""Streamlit demo application for long document summarization."""

import time
from io import BytesIO
from typing import Dict, Optional

import plotly.graph_objects as go
import PyPDF2
import streamlit as st

# Import models
from models.baseline_abstractive import BARTChunkSummarizer
from models.baseline_extractive import LexRankSummarizer, TextRankSummarizer
from models.sliding_window import SlidingWindowSummarizer
from src.faithfulness_checker import FaithfulnessChecker
from src.inference import SummarizationInference

# Page configuration
st.set_page_config(
    page_title="Long Document Summarization",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Load and cache models."""
    models = {
        "TextRank (Extractive)": TextRankSummarizer(num_sentences=5),
        "LexRank (Extractive)": LexRankSummarizer(num_sentences=5),
        "BART Chunks (Abstractive)": BARTChunkSummarizer(),
        "Sliding Window (Abstractive)": SlidingWindowSummarizer(),
    }
    return models


@st.cache_resource
def load_faithfulness_checker():
    """Load faithfulness checker."""
    return FaithfulnessChecker()


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF.

    Args:
        pdf_file: Uploaded PDF file

    Returns:
        Extracted text
    """
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def highlight_text(text: str, summary: str) -> str:
    """Highlight important sentences in the original text.

    Args:
        text: Original text
        summary: Summary text

    Returns:
        HTML with highlighted text
    """
    import nltk

    try:
        sentences = nltk.sent_tokenize(text)
        summary_sentences = nltk.sent_tokenize(summary)
    except LookupError:
        nltk.download("punkt")
        sentences = nltk.sent_tokenize(text)
        summary_sentences = nltk.sent_tokenize(summary)

    # Simple matching (could be improved)
    highlighted = text
    for sent in summary_sentences:
        if sent in text:
            highlighted = highlighted.replace(
                sent, f'<mark style="background-color: #ffeb3b;">{sent}</mark>'
            )

    return highlighted


def create_metrics_chart(metrics: Dict) -> go.Figure:
    """Create metrics visualization.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color="#1f77b4",
        )
    )

    fig.update_layout(
        title="Summary Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400,
    )

    return fig


def main():
    """Main app function."""
    # Header
    st.markdown(
        '<h1 class="main-header">📄 Long Document Summarization System</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    This demo showcases different approaches to summarizing long documents (5K-15K tokens).
    Choose a model, input your text, and get both extractive and abstractive summaries!
    """
    )

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Summarization Model",
        [
            "TextRank (Extractive)",
            "LexRank (Extractive)",
            "BART Chunks (Abstractive)",
            "Sliding Window (Abstractive)",
        ],
    )

    # Parameters
    st.sidebar.subheader("Parameters")

    if "Extractive" in model_name:
        num_sentences = st.sidebar.slider(
            "Number of sentences to extract", min_value=3, max_value=10, value=5
        )
    else:
        max_length = st.sidebar.slider(
            "Maximum summary length (words)",
            min_value=50,
            max_value=500,
            value=256,
        )

    show_faithfulness = st.sidebar.checkbox("Check faithfulness", value=True)
    show_highlights = st.sidebar.checkbox("Highlight important sentences", value=True)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input Document")

        # Input method
        input_method = st.radio(
            "Input method:", ["Paste Text", "Upload File", "Example"]
        )

        input_text = ""

        if input_method == "Paste Text":
            input_text = st.text_area(
                "Enter your document (5K-15K tokens recommended):",
                height=400,
                placeholder="Paste your long document here...",
            )

        elif input_method == "Upload File":
            uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

            if uploaded_file is not None:
                if uploaded_file.type == "application/pdf":
                    input_text = extract_text_from_pdf(uploaded_file)
                else:
                    input_text = uploaded_file.read().decode("utf-8")

                if input_text:
                    st.success("File loaded successfully!")
                    with st.expander("View uploaded text"):
                        st.text(
                            input_text[:1000] + "..."
                            if len(input_text) > 1000
                            else input_text
                        )

        else:  # Example
            input_text = (
                """
            The field of natural language processing (NLP) has undergone a remarkable transformation in recent years,
            driven primarily by the advent of transformer-based architectures and large-scale pre-training. This
            revolution began with the introduction of the Transformer architecture in 2017 by Vaswani et al. in their
            seminal paper "Attention is All You Need." The key innovation was the self-attention mechanism, which
            allows the model to weigh the importance of different words in a sequence when processing each word,
            eliminating the need for recurrence and enabling better parallelization.

            Following this breakthrough, we witnessed the development of BERT (Bidirectional Encoder Representations
            from Transformers) by Google in 2018. BERT used a masked language modeling objective to pre-train deep
            bidirectional representations, achieving state-of-the-art results across multiple NLP benchmarks. Around
            the same time, OpenAI introduced GPT (Generative Pre-trained Transformer), which used autoregressive
            language modeling and demonstrated impressive text generation capabilities.

            The success of these models sparked an explosion of research in transfer learning for NLP. Models like
            RoBERTa, ALBERT, and ELECTRA built upon BERT's foundation, introducing various improvements. Meanwhile,
            the GPT series evolved through GPT-2 and GPT-3, with the latter containing 175 billion parameters and
            demonstrating remarkable few-shot learning capabilities.

            However, standard transformers face a significant limitation: their quadratic complexity with respect to
            sequence length. This makes them computationally expensive for long documents. To address this, researchers
            developed efficient transformer variants. Longformer introduced sparse attention patterns, allowing it to
            process sequences up to 16,384 tokens. BigBird combined random attention, window attention, and global
            attention to achieve similar efficiency gains.

            In the domain of summarization specifically, models like BART and PEGASUS were introduced. BART combines
            a bidirectional encoder with an autoregressive decoder, making it particularly effective for generation
            tasks. PEGASUS was specifically pre-trained for summarization using a gap-sentence generation objective.

            The application of these models to long document summarization remains challenging. Various approaches have
            been proposed, including hierarchical methods that encode paragraphs separately before combining them,
            sliding window techniques that process overlapping chunks, and extract-then-abstract pipelines that first
            select important content before generating summaries.

            Recent work has also focused on evaluation metrics for summarization. While ROUGE scores remain standard,
            researchers have developed additional metrics like BERTScore for semantic similarity and faithfulness
            metrics to detect hallucinations. The field continues to evolve rapidly, with new models and techniques
            emerging regularly, pushing the boundaries of what's possible in natural language understanding and
            generation.
            """
                * 3
            )  # Repeat to make it longer

            st.info("Using example document (click 'Summarize' below)")

    with col2:
        st.subheader("✨ Summary & Analysis")

        if st.button("🚀 Summarize", type="primary", use_container_width=True):
            if not input_text:
                st.error("Please provide input text!")
            else:
                # Show input statistics
                with st.expander("📊 Input Statistics", expanded=True):
                    word_count = len(input_text.split())
                    char_count = len(input_text)

                    stat_cols = st.columns(3)
                    stat_cols[0].metric("Words", f"{word_count:,}")
                    stat_cols[1].metric("Characters", f"{char_count:,}")
                    stat_cols[2].metric("Est. Tokens", f"{word_count * 1.3:.0f}")

                # Load model
                with st.spinner(f"Loading {model_name}..."):
                    try:
                        models = load_models()
                        model = models[model_name]
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        return

                # Generate summary
                with st.spinner("Generating summary..."):
                    start_time = time.time()

                    try:
                        summary = model.summarize(input_text)
                        inference_time = time.time() - start_time

                        # Display summary
                        st.success("Summary generated successfully!")
                        st.markdown("### 📋 Generated Summary")
                        st.write(summary)

                        # Metrics
                        with st.expander("📈 Performance Metrics", expanded=True):
                            summary_words = len(summary.split())
                            compression_ratio = (
                                word_count / summary_words if summary_words > 0 else 0
                            )

                            metric_cols = st.columns(4)
                            metric_cols[0].metric(
                                "Inference Time", f"{inference_time:.2f}s"
                            )
                            metric_cols[1].metric(
                                "Summary Length", f"{summary_words} words"
                            )
                            metric_cols[2].metric(
                                "Compression Ratio", f"{compression_ratio:.1f}x"
                            )
                            metric_cols[3].metric("Type", model_name.split()[0])

                        # Faithfulness check
                        if show_faithfulness:
                            with st.spinner("Checking faithfulness..."):
                                try:
                                    checker = load_faithfulness_checker()
                                    faith_result = checker.check_summary(
                                        input_text, summary
                                    )

                                    st.markdown("### 🔍 Faithfulness Analysis")

                                    faith_cols = st.columns(3)
                                    faith_cols[0].metric(
                                        "Faithfulness Score",
                                        f"{faith_result['overall_score']:.2%}",
                                    )
                                    faith_cols[1].metric(
                                        "Sentences Checked",
                                        faith_result["num_sentences"],
                                    )
                                    faith_cols[2].metric(
                                        "Potential Hallucinations",
                                        len(faith_result["hallucinations"]),
                                    )

                                    if faith_result["hallucinations"]:
                                        with st.expander("⚠️ Potential Hallucinations"):
                                            for h in faith_result["hallucinations"]:
                                                st.warning(
                                                    f"**Score: {h['score']:.2f}** - {h['sentence']}"
                                                )

                                except Exception as e:
                                    st.warning(f"Could not check faithfulness: {e}")

                        # Highlights
                        if show_highlights:
                            with st.expander("💡 Highlighted Source Text"):
                                highlighted = highlight_text(input_text, summary)
                                st.markdown(highlighted, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
                        st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center;">
        <p>Long Document Summarization System | NLP Final Project</p>
        <p>Supports documents up to 16K tokens | Multiple architectural approaches</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
