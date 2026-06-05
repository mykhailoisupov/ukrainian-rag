"""
Active RAG Strategies for Ukrainian RAG Pipeline.

This package implements agentic RAG techniques from RAG-from-scratch
Notebooks 15-18:

- crag: Corrective RAG — grades retrieved docs, falls back to web search
  (uses LangGraph because the decision flow is naturally a graph)
- self_rag: Self-RAG — decides whether to retrieve, grades relevance
  and hallucination (hand-rolled because the logic is a simple linear loop)
- adaptive_rag: Adaptive RAG — classifies query complexity and routes
  to the appropriate strategy (uses LangGraph for multi-path routing)
"""
