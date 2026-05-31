# Ukrainian RAG System

A modular Retrieval-Augmented Generation (RAG) pipeline tailored for processing and querying Ukrainian language text corpora with standardized RAGAS evaluation.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-v0.2.1-orange.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-brightgreen.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-v0.5.0-blueviolet.svg)
![RAGAS](https://img.shields.io/badge/RAGAS-v0.1.9-red.svg)

## Architecture

The diagram below outlines the logical flow of document ingestion, vector indexing, query retrieval, and LLM grounding:

```text
                      [ Raw Wikipedia Text ]
                                 │
                                 ▼ (Recursive Splitter)
                             [ Chunks ]
                                 │
                                 ▼ (sentence-transformers)
                           [ Embeddings ]
                                 │
                                 ▼ (Index)
                        [( ChromaDB Vector Store )]
                                 ▲
                                 │ Similarity Search
                                 │
      User Query ────► [ embed_query() ] ─────► Context Chunks
                                                    │
                                                    ▼
      User Answer ◄─── [ GPT-4o-Mini ] ◄────────────┘ (Prompt Injection)
```

## Tech Stack

The architecture of this repository consists of the following components:

*   **Language**: Python (v3.10+)
*   **Orchestration**: LangChain
*   **Vector Database**: ChromaDB (configured with persistent local storage)
*   **Local Embeddings**: sentence-transformers (using the multilingual 'paraphrase-multilingual-MiniLM-L12-v2' model)
*   **Generator LLM**: OpenAI API (using the 'gpt-4o-mini' model)
*   **Evaluation Framework**: RAGAS (calculating faithfulness, answer relevancy, and context recall)

## Evaluation Results

The pipeline's retrieval and synthesis capabilities were assessed against a benchmark of 10 Ukrainian test questions using RAGAS. The scores obtained are as follows:

| Metric               | Score  | Explanation                                                                                                                                    |
| :---------------------| :-------| :-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Faithfulness**     | 0.7500 | Measures the factual consistency of the generated answer against the retrieved context, ensuring the response contains no hallucinations.      |
| **Answer Relevancy** | 0.5324 | Measures how directly and appropriately the generated answer addresses the user's initial query, penalizing incomplete or redundant responses. |
| **Context Recall**   | 0.7000 | Measures retrieval accuracy by checking whether all key ground-truth facts are present in the retrieved chunks.                                |

## Future Work

To further optimize the performance of the Ukrainian RAG pipeline, the following areas will be explored:

*   **Fine-Tuning Domain-Specific Embeddings**: Fine-tune the dense vector embeddings on a large-scale, custom Ukrainian corpus to improve semantic similarity performance.
*   **Integrating Hybrid Retrieval**: Combine dense semantic retrieval with sparse lexical retrieval (such as BM25) to better handle specific morphosyntactic structures of the Ukrainian language.
*   **Exploring Open-Source LLMs**: Benchmark the pipeline using high-performing open-source multilingual LLMs (e.g., Llama 3 or Mistral models) to enable fully local deployment.

## Author

*   **Mykhailo Isupov**
*   GitHub: [mykhailoisupov](https://github.com/mykhailoisupov)
