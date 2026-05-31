import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

class RAGPipeline:
    def __init__(self):
        client = chromadb.PersistentClient(path='data/processed/')
        self.collection = client.get_or_create_collection(name='ukrainian_rag')
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.openai_model = 'gpt-4o-mini'

    def embed_query(self, query: str) -> list:
        return self.encoder.encode(query).tolist()

    def retrieve(self, query: str, n_results: int = 3) -> list[dict]:
        query_embedding = self.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        retrieved = []
        if results and results.get('documents') and len(results['documents']) > 0:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            for doc, meta, dist in zip(documents, metadatas, distances):
                retrieved.append({
                    'text': doc,
                    'source': meta.get('source'),
                    'chunk_index': meta.get('chunk_index'),
                    'distance': dist
                })
        return retrieved

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        context_str = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_str += f"[{i}] (Джерело: {chunk['source']}): {chunk['text']}\n\n"
            
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Ти корисний асистент, який відповідає на питання виключно на основі наданого контексту українською мовою. Якщо відповідь не міститься в контексті, скажи: 'Я не можу знайти цю інформацію в наданих документах.'"
                    },
                    {
                        "role": "user", 
                        "content": f"Контекст:\n{context_str}\n\nПитання: {query}"
                    }
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Помилка OpenAI API: {e}"

    def query(self, question: str, n_results: int = 3) -> dict:
        context_chunks = self.retrieve(question, n_results=n_results)
        answer = self.generate(question, context_chunks)
        return {
            'question': question,
            'answer': answer,
            'context_chunks': context_chunks,
            'n_chunks_used': len(context_chunks)
        }

if __name__ == '__main__':
    pipeline = RAGPipeline()
    
    questions = [
        "Хто такий Тарас Шевченко?",
        "Що таке штучний інтелект?",
        "Коли сталася аварія на Чорнобильській АЕС?"
    ]
    
    for q in questions:
        print("=" * 80)
        print(f"ПИТАННЯ: {q}")
        print("=" * 80)
        res = pipeline.query(q)
        print(f"ВІДПОВІДЬ:\n{res['answer']}\n")
        print("Використані фрагменти контексту:")
        for idx, chunk in enumerate(res['context_chunks'], 1):
            print(f"  {idx}. [{chunk['source']} (chunk {chunk['chunk_index']})] (distance: {chunk['distance']:.4f})")
            print(f"     Text snippet: {chunk['text'][:120].strip()}...")
        print("\n")
