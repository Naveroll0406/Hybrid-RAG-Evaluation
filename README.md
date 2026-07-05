# Hybrid RAG Pipeline with LangSmith & RAGAS

This repository contains a highly optimized Retrieval-Augmented Generation (RAG) pipeline built to extract and answer questions from complex PDF documents with maximum accuracy.

## Architecture

The pipeline moves beyond basic vector search by implementing a **Hybrid Retrieval** system:
- **Multi-Query Generation**: Synthesizes multiple perspectives of the user's question.
- **Hybrid Search**: Combines Dense Retrieval (FAISS) for semantic understanding and Sparse Retrieval (BM25) for exact keyword matching.
- **Reciprocal Rank Fusion (RRF)**: Merges the dense and sparse results intelligently.
- **Cross-Encoder Reranking**: Uses `BAAI/bge-reranker-base` to aggressively filter chunks and ensure only the absolute most relevant context reaches the generation step.

## Evaluation & Optimization

This pipeline was rigorously evaluated using **RAGAS (LLM-as-a-judge)** and traced with **LangSmith**. Through an iterative process of analyzing traces and tuning parameters (such as chunk size, overlap, `top_n`, and `k`), we achieved the following metrics on a custom benchmark dataset:

- 🟢 **Faithfulness**: ~96% (Virtually zero hallucination)
- 🟢 **Answer Relevancy**: ~86% (Highly concise and direct answers)
- 🟢 **Context Recall**: ~83%
- 🟢 **Context Precision**: ~81%

## Files
- `Hybrid_RAG.py`: The ingestion script that parses PDFs, chunks the text, builds the FAISS/BM25 indexes, and sets up the retrieval logic.
- `evaluate_rag.py`: The evaluation script that loads the indexes, defines the generation prompt, executes the RAG pipeline, and grades it using RAGAS.

## Results

Here are the final RAGAS evaluation scores proving the pipeline's effectiveness:

```text
============================================================
📊 RAGAS EVALUATION RESULTS
============================================================

🏆 OVERALL SCORES:
----------------------------------------
  faithfulness              ███████████████████░ 0.9688
  answer_relevancy          █████████████████░░░ 0.8628
  context_precision         ████████████████░░░░ 0.8171
  context_recall            ████████████████░░░░ 0.8333
```

*(Execution traces and breakdowns are available in the LangSmith dashboard).*

### LangSmith Execution Trace
You can view a live, interactive execution trace of this pipeline directly in LangSmith:
🔗 **[View LangSmith Trace (RAGAS Evaluation)](https://smith.langchain.com/public/0f9486a0-bd8e-478f-a528-697b9e7d1934/r/019f32d0-e5ea-7cc3-b5d5-eef19ce41967)**
