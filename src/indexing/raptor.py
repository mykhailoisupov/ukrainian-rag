"""
RAPTOR Indexing for Ukrainian RAG Pipeline.

From RAG-from-scratch Notebook 13:

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
builds a hierarchical index by recursively clustering and summarizing
document chunks:

1. Start with leaf-level chunks (from RecursiveCharacterTextSplitter)
2. Embed all chunks
3. Cluster them using Gaussian Mixture Models (GMM)
4. Summarize each cluster into a parent node using the LLM
5. Recursively repeat steps 2-4 on the summaries
6. Store ALL levels in ChromaDB (leaves + summaries at every level)

Why this helps: Questions that require *high-level* understanding
(e.g. "What is the overall significance of Chornobyl?") are better
answered by summary nodes, while specific questions still match leaves.
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()


def _get_openai_embeddings(texts: list[str], client: OpenAI,
                           model: str = 'text-embedding-3-small') -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _summarize_cluster(texts: list[str], client: OpenAI,
                       model: str = 'gpt-4o-mini') -> str:
    """Summarize a cluster of text chunks into a single parent node."""
    combined = "\n\n---\n\n".join(texts)
    if len(combined) > 8000:
        combined = combined[:8000] + "..."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ти — асистент для узагальнення інформації. Створи детальне "
                    "резюме (3-5 речень) наданих текстових фрагментів українською "
                    "мовою. Збережи ключові факти, дати та імена."
                ),
            },
            {"role": "user", "content": combined},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def _cluster_embeddings(embeddings: np.ndarray, n_clusters: int = None) -> list[list[int]]:
    """
    Cluster embeddings using Gaussian Mixture Models (GMM).

    Returns a list of clusters, where each cluster is a list of indices.
    """
    from sklearn.mixture import GaussianMixture

    n_samples = len(embeddings)
    if n_samples <= 1:
        return [list(range(n_samples))]

    if n_clusters is None:
        n_clusters = max(2, min(n_samples // 5, 10))

    n_clusters = min(n_clusters, n_samples)

    if n_clusters <= 1:
        return [list(range(n_samples))]

    try:
        from umap import UMAP
        if n_samples > 10 and embeddings.shape[1] > 10:
            n_components = min(10, n_samples - 1)
            reducer = UMAP(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        else:
            reduced = embeddings
    except ImportError:
        reduced = embeddings

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(reduced)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    return list(clusters.values())


def ingest_raptor(max_levels: int = 3):
    """
    RAPTOR hierarchical ingestion pipeline.

    Args:
        max_levels: Maximum number of recursive summarization levels.
    """
    raw_dir = os.path.join("data", "raw")
    file_pattern = os.path.join(raw_dir, "*.txt")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No .txt files found in {raw_dir}. Please run the data loader first.")
        return

    print(f"[RAPTOR] Found {len(file_paths)} files for ingestion.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
    )

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embedding_model = 'text-embedding-3-small'

    print("\nLevel 0: Creating leaf chunks...")
    all_texts = []
    all_metadata = []
    all_ids = []

    for file_path in tqdm(file_paths, desc="Files"):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadata.append({
                    "source": filename,
                    "chunk_index": i,
                    "level": 0,
                    "indexing": "raptor",
                })
                all_ids.append(f"raptor_L0_{filename}_chunk_{i}")
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print(f"Level 0: {len(all_texts)} leaf chunks")

    current_texts = list(all_texts)

    for level in range(1, max_levels + 1):
        if len(current_texts) <= 3:
            print(f"Level {level}: Too few texts ({len(current_texts)}) to cluster. Stopping.")
            break

        print(f"\nLevel {level}: Clustering {len(current_texts)} texts...")

        embeddings = []
        batch_size = 50
        for i in range(0, len(current_texts), batch_size):
            batch = current_texts[i:i + batch_size]
            embeddings.extend(_get_openai_embeddings(batch, openai_client, embedding_model))
        embeddings_array = np.array(embeddings)

        clusters = _cluster_embeddings(embeddings_array)
        print(f"Level {level}: Created {len(clusters)} clusters")

        level_summaries = []
        for cluster_idx, indices in enumerate(tqdm(clusters, desc=f"Summarizing L{level}")):
            cluster_texts = [current_texts[i] for i in indices]
            summary = _summarize_cluster(cluster_texts, openai_client)
            level_summaries.append(summary)

            all_texts.append(summary)
            all_metadata.append({
                "source": "raptor_summary",
                "chunk_index": cluster_idx,
                "level": level,
                "indexing": "raptor",
                "cluster_size": len(indices),
            })
            all_ids.append(f"raptor_L{level}_cluster_{cluster_idx}")

        current_texts = level_summaries

    total = len(all_texts)
    print(f"\nTotal nodes (all levels): {total}")

    client = chromadb.PersistentClient(path='data/processed/')
    collection = client.get_or_create_collection(name='ukrainian_rag_raptor')

    batch_size = 50
    print(f"Embedding and storing all nodes in batches of {batch_size}...")

    for i in tqdm(range(0, total, batch_size), desc="Ingesting"):
        end = min(i + batch_size, total)
        batch_embeddings = _get_openai_embeddings(
            all_texts[i:end], openai_client, embedding_model
        )
        collection.add(
            ids=all_ids[i:end],
            embeddings=batch_embeddings,
            metadatas=all_metadata[i:end],
            documents=all_texts[i:end],
        )

    print(f"\n[RAPTOR] Ingestion complete. {collection.count()} nodes in DB.")
