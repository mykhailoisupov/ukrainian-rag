# Ukrainian RAG System

A modular Retrieval-Augmented Generation (RAG) pipeline tailored for processing and querying Ukrainian language text corpora. Implements **18 advanced RAG techniques** from the [RAG-from-scratch](https://github.com/langchain-ai/rag-from-scratch) series, adapted for Ukrainian with standardized RAGAS evaluation.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-v0.2.1-orange.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-brightgreen.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-v0.5.0-blueviolet.svg)
![RAGAS](https://img.shields.io/badge/RAGAS-v0.1.9-red.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-v0.1.0-yellow.svg)

## Architecture

The diagram below outlines the full pipeline — from ingestion through advanced retrieval to generation:

```text
                      [ Raw Wikipedia Text ]
                                │
                                ▼ (Recursive Splitter)
                            [ Chunks ]
                                │
                    ┌───────────┼───────────────┐
                    ▼           ▼               ▼
              [ Flat Index ] [ Multi-Repr ] [ RAPTOR ]   [ ColBERT ]
              (ChromaDB)     (Summary→Embed) (Hierarchical) (Per-token)
                    │           │               │            │
                    └───────────┼───────────────┘            │
                                ▼                            ▼
                    [( ChromaDB Vector Store )]    [ ColBERT Index ]
                                ▲
                                │ Similarity Search
                                │
  User Query ──►[ Query Translation ]──►[ Routing ]──► Context Chunks
                  │                       │                  │
                  ├─ Multi-Query          ├─ Logical         ▼
                  ├─ RAG-Fusion           └─ Semantic  [ Active RAG ]
                  ├─ Decomposition                       │
                  ├─ Step-back                           ├─ CRAG
                  └─ HyDE                                ├─ Self-RAG
                                                         └─ Adaptive RAG
                                                              │
  User Answer ◄─── [ GPT-4o-Mini ] ◄──────────────────────────┘
```

## Tech Stack

*   **Language**: Python (v3.10+)
*   **Orchestration**: LangChain, LangGraph
*   **Vector Database**: ChromaDB (configured with persistent local storage)
*   **Embeddings**: OpenAI `text-embedding-3-small`
*   **Generator LLM**: OpenAI API (`gpt-4o-mini`)
*   **Evaluation Framework**: RAGAS (faithfulness, answer relevancy, context recall)
*   **Web Search**: Tavily API (for CRAG fallback)
*   **ColBERT**: RAGatouille (optional, for late-interaction retrieval)

## Implemented Techniques

### Query Translation (Notebooks 5–9)

| # | Technique | Description |
|---|-----------|-------------|
| 5 | **Multi-Query** | Generates N alternative phrasings via LLM, retrieves for each, unions results |
| 6 | **RAG-Fusion** | Multi-Query + Reciprocal Rank Fusion (RRF) re-ranking |
| 7 | **Decomposition** | Breaks complex questions into 2–4 sub-questions |
| 8 | **Step-back** | Abstracts the question for broader context retrieval |
| 9 | **HyDE** | Generates a hypothetical answer, embeds it, retrieves by similarity |

### Routing (Notebooks 10–11)

| # | Technique | Description |
|---|-----------|-------------|
| 10 | **Logical Routing** | LLM classifies question type → selects optimal prompt template |
| 11 | **Semantic Routing** | Cosine similarity between query and prompt description embeddings |

### Advanced Indexing (Notebooks 12–14)

| # | Technique | Description |
|---|-----------|-------------|
| 12 | **Multi-Representation** | Embed LLM summaries for retrieval, store full chunks for generation |
| 13 | **RAPTOR** | Recursive clustering (GMM + UMAP) and hierarchical summarization |
| 14 | **ColBERT** | Per-token late-interaction retrieval via RAGatouille |

### Active RAG (Notebooks 15–18)

| # | Technique | Framework | Rationale |
|---|-----------|-----------|-----------|
| 15 | **CRAG** | LangGraph | The decision flow has genuine branching (web search vs. direct generation), which maps naturally to a directed graph with conditional edges |
| 16 | **Self-RAG** | Hand-rolled | The flow is a linear retry loop (retrieve → grade → generate → check → retry), not a graph — a simple `while` loop is more readable and debuggable |
| 17–18 | **Adaptive RAG** | LangGraph | Classifies query complexity and routes to 3 fundamentally different execution paths — exactly what graphs are designed for |

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Wikipedia article fetcher
│   ├── ingestion.py            # Chunking + embedding + ChromaDB indexing
│   ├── rag_pipeline.py         # Main RAGPipeline class (single entry point)
│   ├── query_translation.py    # 5 query translation strategies
│   ├── routing.py              # Logical + semantic routing
│   ├── evaluation.py           # Multi-strategy RAGAS evaluation
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── multi_representation.py
│   │   ├── raptor.py
│   │   └── colbert.py
│   └── active_rag/
│       ├── __init__.py
│       ├── crag.py             # LangGraph
│       ├── self_rag.py         # Hand-rolled
│       └── adaptive_rag.py     # LangGraph
├── data/
│   ├── raw/                    # Ukrainian Wikipedia .txt files
│   ├── processed/              # ChromaDB persistent storage
│   └── evaluation_results.json
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
├── .env                        # OPENAI_API_KEY, TAVILY_API_KEY
└── README.md
```

## Usage

### Basic Query
```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.query("Хто такий Тарас Шевченко?")
print(result['answer'])
```

### With Query Translation
```python
# Available strategies: vanilla, multi_query, rag_fusion, decompose, step_back, hyde
result = pipeline.query("Хто такий Тарас Шевченко?", strategy="hyde")
```

### With Routing
```python
result = pipeline.query("Хто такий Тарас Шевченко?", use_routing=True)
```

### With Active RAG
```python
# Available modes: crag, self_rag, adaptive
result = pipeline.query("Як Голодомор вплинув на демографію?", active_rag="adaptive")
```

### Multi-Strategy Evaluation
```bash
# Evaluate specific strategies
python -m src.evaluation --strategies vanilla hyde multi_query

# Evaluate active RAG modes
python -m src.evaluation --strategies vanilla --active-rag crag self_rag

# Full benchmark
python -m src.evaluation --strategies vanilla multi_query rag_fusion decompose step_back hyde --active-rag crag self_rag adaptive
```

## Evaluation Results

Baseline RAGAS scores (strategy: `vanilla`):

| Metric               | Score  | Explanation |
| :---------------------| :-------| :-----------|
| **Faithfulness**     | 0.7500 | Measures factual consistency of the generated answer against retrieved context |
| **Answer Relevancy** | 0.5324 | Measures how directly the answer addresses the user's query |
| **Context Recall**   | 0.7000 | Measures whether all key ground-truth facts are present in retrieved chunks |

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/mykhailoisupov/ukrainian-rag.git
cd ukrainian-rag

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file with:
#   OPENAI_API_KEY=your_key_here
#   TAVILY_API_KEY=your_key_here  (optional, for CRAG)

# 5. Download Ukrainian Wikipedia articles
python -m src.data_loader

# 6. Ingest documents into ChromaDB
python -m src.ingestion

# 7. Run a query
python -m src.rag_pipeline
```

## Author

*   **Mykhailo Isupov**
*   GitHub: [mykhailoisupov](https://github.com/mykhailoisupov)
