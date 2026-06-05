"""
ColBERT-style Indexing for Ukrainian RAG Pipeline.

From RAG-from-scratch Notebook 14:

ColBERT (Contextualized Late Interaction over BERT) uses a fundamentally
different retrieval approach than single-vector embeddings:

Standard approach:
    Document → single vector (384-1536 dims) → cosine similarity with query vector

ColBERT approach:
    Document → per-TOKEN embeddings (N_doc × 128 dims)
    Query → per-TOKEN embeddings (N_query × 128 dims)
    Score = sum of MaxSim(query_token_i, all_doc_tokens) for each query token

Why this helps: Late interaction preserves fine-grained token-level semantics.
A query token like "Шевченко" can match precisely against the token "Шевченко"
in documents, even if the surrounding context differs. This is especially
powerful for Ukrainian — a morphologically rich language where word forms
vary heavily (відмінки, дієвідміни).

This module uses the RAGatouille library for ColBERT indexing/retrieval.

Note: RAGatouille requires PyTorch and downloads a ColBERT model (~400MB)
on first use. This is the heaviest dependency in the project.
"""

import os
import sys
import glob
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()


def ingest_colbert(index_name: str = 'ukrainian_rag_colbert'):
    """
    ColBERT ingestion pipeline using RAGatouille.

    Creates a ColBERT index from raw Ukrainian text documents.
    The index is stored locally and can be queried via search_colbert().

    Args:
        index_name: Name for the ColBERT index.
    """
    try:
        from ragatouille import RAGPretrainedModel
    except ImportError:
        print(
            "ERROR: RAGatouille is not installed.\n"
            "Install it with: pip install ragatouille\n"
            "Note: This also requires PyTorch."
        )
        return

    raw_dir = os.path.join("data", "raw")
    file_pattern = os.path.join(raw_dir, "*.txt")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No .txt files found in {raw_dir}. Please run the data loader first.")
        return

    print(f"[ColBERT] Found {len(file_paths)} files for ingestion.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []
    all_doc_ids = []
    all_metadata = []

    print("\nChunking documents...")
    for file_path in tqdm(file_paths, desc="Files"):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_doc_ids.append(f"colbert_{filename}_chunk_{i}")
                all_metadata.append({
                    "source": filename,
                    "chunk_index": i,
                    "indexing": "colbert",
                })
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    total = len(all_chunks)
    print(f"Total chunks: {total}")

    if total == 0:
        print("No chunks to index.")
        return

    print("\nInitializing ColBERT model (this may download ~400MB on first run)...")
    rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    print(f"\nBuilding ColBERT index '{index_name}'...")
    index_path = os.path.join("data", "processed", "colbert_index")
    os.makedirs(index_path, exist_ok=True)

    rag.index(
        collection=all_chunks,
        document_ids=all_doc_ids,
        document_metadatas=all_metadata,
        index_name=index_name,
        max_document_length=512,
        split_documents=False,
    )

    print(f"\n[ColBERT] Indexing complete. {total} chunks indexed.")
    print(f"Index stored at: {index_path}")


def search_colbert(query: str, n_results: int = 3,
                   index_name: str = 'ukrainian_rag_colbert') -> list[dict]:
    """
    Search the ColBERT index for relevant chunks.

    Args:
        query: The search query in Ukrainian.
        n_results: Number of results to return.
        index_name: Name of the ColBERT index to search.

    Returns:
        List of dicts with 'text', 'source', 'chunk_index', 'distance' keys,
        matching the same format as RAGPipeline.retrieve().
    """
    try:
        from ragatouille import RAGPretrainedModel
    except ImportError:
        raise ImportError("RAGatouille is not installed. pip install ragatouille")

    rag = RAGPretrainedModel.from_index(index_name)
    results = rag.search(query, k=n_results)

    retrieved = []
    for result in results:
        retrieved.append({
            'text': result.get('content', ''),
            'source': result.get('document_metadata', {}).get('source', 'unknown'),
            'chunk_index': result.get('document_metadata', {}).get('chunk_index', -1),
            'distance': 1.0 - result.get('score', 0.0),
        })

    return retrieved
