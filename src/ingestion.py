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


def get_openai_embeddings(texts: list[str], client: OpenAI, model: str = 'text-embedding-3-small') -> list[list[float]]:
    """
    Generate embeddings for a list of texts using OpenAI's embedding API.
    Handles batching internally — OpenAI supports up to 2048 texts per request.
    """
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def ingest_documents(indexing_strategy: str = 'flat'):
    """
    Loads raw Ukrainian text documents, chunks them using RecursiveCharacterTextSplitter,
    generates embeddings using OpenAI, and indexes them in ChromaDB.

    Args:
        indexing_strategy: The indexing strategy to use. Options:
            - 'flat': Standard flat index (default)
            - 'multi_repr': Multi-representation indexing
            - 'raptor': RAPTOR hierarchical indexing
            - 'colbert': ColBERT late-interaction indexing
    """
    if indexing_strategy != 'flat':
        if indexing_strategy == 'multi_repr':
            from src.indexing.multi_representation import ingest_multi_representation
            return ingest_multi_representation()
        elif indexing_strategy == 'raptor':
            from src.indexing.raptor import ingest_raptor
            return ingest_raptor()
        elif indexing_strategy == 'colbert':
            from src.indexing.colbert import ingest_colbert
            return ingest_colbert()
        else:
            raise ValueError(f"Unknown indexing strategy: {indexing_strategy}")

    raw_dir = os.path.join("data", "raw")
    file_pattern = os.path.join(raw_dir, "*.txt")
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No .txt files found in {raw_dir}. Please run the data loader first.")
        return
    
    print(f"Found {len(file_paths)} files for ingestion.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embedding_model = 'text-embedding-3-small'
    print(f"Using OpenAI embedding model: '{embedding_model}'")
    
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    print("\nProcessing files:")
    pbar = tqdm(file_paths, desc="Files processed")
    for file_path in pbar:
        filename = os.path.basename(file_path)
        pbar.set_description(f"Processing {filename}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": filename,
                    "chunk_index": i
                })
                all_ids.append(f"{filename}_chunk_{i}")
                
        except Exception as e:
            print(f"\nError processing file {filename}: {e}")
            
    total_chunks = len(all_chunks)
    print(f"\nTotal chunks created: {total_chunks}")
    
    if total_chunks == 0:
        print("No chunks to index.")
        return
        
    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path='data/processed/')
    collection = client.get_or_create_collection(name='ukrainian_rag')
    
    batch_size = 50
    print(f"Generating OpenAI embeddings and adding to ChromaDB in batches of {batch_size}...")
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Ingesting batches"):
        end_idx = min(i + batch_size, total_chunks)
        
        batch_chunks = all_chunks[i:end_idx]
        batch_metadata = all_metadata[i:end_idx]
        batch_ids = all_ids[i:end_idx]
        
        batch_embeddings = get_openai_embeddings(batch_chunks, openai_client, embedding_model)
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadata,
            documents=batch_chunks
        )
        
    total_docs_in_db = collection.count()
    print("\n" + "=" * 50)
    print("Ingestion Completed Successfully!")
    print("=" * 50)
    print(f"Total chunks created: {total_chunks}")
    print(f"Total documents in ChromaDB: {total_docs_in_db}")
    print(f"Embedding model: {embedding_model}")
    print("=" * 50)

if __name__ == '__main__':
    ingest_documents()