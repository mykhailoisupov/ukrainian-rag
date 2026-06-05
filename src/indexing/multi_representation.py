"""
Multi-Representation Indexing for Ukrainian RAG Pipeline.

From RAG-from-scratch Notebook 12:

The key insight is that the text used for *retrieval* doesn't have to be the
same text used for *generation*. This strategy:

1. Takes each raw text chunk
2. Generates a concise LLM summary of that chunk
3. Stores the SUMMARY embedding in ChromaDB (used for retrieval)
4. Stores the FULL ORIGINAL CHUNK as the document (used for generation)

Why this helps: Summaries capture the core semantic meaning more densely,
so embedding-based retrieval on summaries often outperforms retrieval on
raw chunks — especially for long, noisy chunks.
"""

import os
import sys
import glob
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


def _summarize_chunk(chunk: str, client: OpenAI, model: str = 'gpt-4o-mini') -> str:
    """Generate a concise Ukrainian-language summary of a text chunk."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ти — асистент для створення стислих резюме. "
                    "Створи коротке резюме (1-2 речення) наданого тексту "
                    "українською мовою. Збережи ключові факти та деталі."
                ),
            },
            {"role": "user", "content": chunk},
        ],
        temperature=0.0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def _get_openai_embeddings(texts: list[str], client: OpenAI,
                           model: str = 'text-embedding-3-small') -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def ingest_multi_representation():
    """
    Multi-representation ingestion pipeline:
    - Chunk raw documents
    - Summarize each chunk via LLM
    - Embed the SUMMARIES
    - Store in ChromaDB with full chunks as documents
    """
    raw_dir = os.path.join("data", "raw")
    file_pattern = os.path.join(raw_dir, "*.txt")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No .txt files found in {raw_dir}. Please run the data loader first.")
        return

    print(f"[Multi-Repr] Found {len(file_paths)} files for ingestion.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
    )

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embedding_model = 'text-embedding-3-small'

    all_chunks = []
    all_summaries = []
    all_metadata = []
    all_ids = []

    print("\nProcessing files and generating summaries:")
    for file_path in tqdm(file_paths, desc="Files"):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = text_splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                summary = _summarize_chunk(chunk, openai_client)
                all_chunks.append(chunk)
                all_summaries.append(summary)
                all_metadata.append({
                    "source": filename,
                    "chunk_index": i,
                    "summary": summary,
                    "indexing": "multi_repr",
                })
                all_ids.append(f"multi_repr_{filename}_chunk_{i}")

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    total = len(all_chunks)
    print(f"\nTotal chunks created: {total}")

    if total == 0:
        print("No chunks to index.")
        return

    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path='data/processed/')
    collection = client.get_or_create_collection(name='ukrainian_rag_multi_repr')

    batch_size = 50
    print(f"Embedding SUMMARIES and storing FULL CHUNKS in batches of {batch_size}...")

    for i in tqdm(range(0, total, batch_size), desc="Ingesting"):
        end = min(i + batch_size, total)

        batch_embeddings = _get_openai_embeddings(
            all_summaries[i:end], openai_client, embedding_model
        )

        collection.add(
            ids=all_ids[i:end],
            embeddings=batch_embeddings,
            metadatas=all_metadata[i:end],
            documents=all_chunks[i:end],
        )

    print(f"\n[Multi-Repr] Ingestion complete. {collection.count()} documents in DB.")
