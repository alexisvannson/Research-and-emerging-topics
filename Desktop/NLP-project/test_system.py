"""Quick end-to-end test of the summarization system."""

import sys

print("Long Document Summarization System - Quick Test")
 

# Test document
test_doc = """
Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language. In particular, how to program computers to process and analyze
large amounts of natural language data. The goal is a computer capable of
understanding the contents of documents, including the contextual nuances of
the language within them.

The technology can then accurately extract information and insights contained in
the documents as well as categorize and organize the documents themselves.
Challenges in natural language processing frequently involve speech recognition,
natural language understanding, and natural language generation.

Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing
published an article titled "Computing Machinery and Intelligence" which proposed
what is now called the Turing test as a criterion of intelligence. The Georgetown
experiment in 1954 involved fully automatic translation of more than sixty Russian
sentences into English.
""" * 20  # Repeat to make it longer

print(f"\nTest document: {len(test_doc.split())} words")
 

# Test 1: TextRank
print("\n1. Testing TextRank (Extractive)...")
try:
    from models.baseline_extractive import TextRankSummarizer

    textrank = TextRankSummarizer(num_sentences=3)
    summary = textrank.summarize(test_doc)

    print(f"   ✓ TextRank works!")
    print(f"   Summary length: {len(summary.split())} words")
    print(f"   First 100 chars: {summary[:100]}...")
except Exception as e:
    print(f"   ✗ TextRank failed: {e}")
    sys.exit(1)

# Test 2: LexRank
print("\n2. Testing LexRank (Extractive)...")
try:
    from models.baseline_extractive import LexRankSummarizer

    lexrank = LexRankSummarizer(num_sentences=3)
    summary = lexrank.summarize(test_doc)

    print(f"   ✓ LexRank works!")
    print(f"   Summary length: {len(summary.split())} words")
    print(f"   First 100 chars: {summary[:100]}...")
except Exception as e:
    print(f"   ✗ LexRank failed: {e}")
    sys.exit(1)

# Test 3: Sliding Window
print("\n3. Testing Sliding Window (Abstractive)...")
try:
    from models.sliding_window import SlidingWindowSummarizer

    info_only = SlidingWindowSummarizer(window_size=512, overlap_size=128)
    window_info = info_only.get_window_info(test_doc)

    print(f"   ✓ Sliding Window works!")
    print(f"   Total tokens: {window_info['total_tokens']}")
    print(f"   Number of windows: {window_info['num_windows']}")
    print(f"   Note: Actual summarization requires large models (skipping for quick test)")
except Exception as e:
    print(f"   ✗ Sliding Window failed: {e}")
    sys.exit(1)

# Test 4: Utilities
print("\n4. Testing Utilities...")
try:
    from models.utils import (
        set_seed,
        get_device,
        format_time,
        compute_compression_ratio,
        AverageMeter
    )

    set_seed(42)
    device = get_device()
    time_str = format_time(125.5)
    ratio = compute_compression_ratio(test_doc, summary)
    meter = AverageMeter()
    meter.update(10)

    print(f"   ✓ Utilities work!")
    print(f"   Device: {device}")
    print(f"   Compression ratio: {ratio:.2f}x")
except Exception as e:
    print(f"   ✗ Utilities failed: {e}")
    sys.exit(1)

# Test 5: Inference interface
print("\n5. Testing Inference Interface...")
try:
    from src.inference import SummarizationInference

    inference = SummarizationInference("textrank", {"extractive": {"num_sentences": 2}})
    result = inference.summarize(test_doc, return_metrics=True)

    print(f"   ✓ Inference interface works!")
    print(f"   Inference time: {result['metrics']['inference_time']:.3f}s")
    print(f"   Compression: {result['metrics']['compression_ratio']:.2f}x")
except Exception as e:
    print(f"   ✗ Inference interface failed: {e}")
    sys.exit(1)

print("\n" + " 
print("✓ All tests passed! System is working correctly.")
 
print("\nNext steps:")
print("  - Run 'make test' for full unit tests")
print("  - Run 'make demo' to launch the interactive demo")
print("  - Run 'make download-data' to get datasets")
 
