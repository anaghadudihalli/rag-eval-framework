"""RAG Evaluation & Observability Framework.

An automated pipeline for measuring and monitoring RAG system quality:
- Retrieval quality (precision@K, recall@K)
- Semantic similarity (sentence-transformers cosine)
- Hallucination rate (LLM-as-judge via GPT-3.5-turbo)
- Confidence-score routing correctness
"""

__version__ = "0.1.0"
__author__ = "Anagha Dudihalli"
