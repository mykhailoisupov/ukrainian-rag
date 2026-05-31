import os
import sys
import glob
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def ingest_documents():
    """
    Loads raw Ukrainian text documents, chunks them using RecursiveCharacterTextSplitter,
    generates embeddings using SentenceTransformer, and indexes them in ChromaDB.
    """
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
    
    print("Loading SentenceTransformer model ('paraphrase-multilingual-MiniLM-L12-v2')...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
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
    print(f"Adding documents to ChromaDB in batches of {batch_size}...")
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Ingesting batches"):
        end_idx = min(i + batch_size, total_chunks)
        
        batch_chunks = all_chunks[i:end_idx]
        batch_metadata = all_metadata[i:end_idx]
        batch_ids = all_ids[i:end_idx]
        
        batch_embeddings = model.encode(batch_chunks).tolist()
        
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
    print("=" * 50)

if __name__ == '__main__':
    ingest_documents()