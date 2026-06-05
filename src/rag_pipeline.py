import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        db_path = os.path.join(project_root, 'data', 'processed')
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_or_create_collection(name='ukrainian_rag')
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.openai_model = 'gpt-4o-mini'
        self.embedding_model = 'text-embedding-3-small'

    def embed_query(self, query: str) -> list:
        """Generate embedding for a query using OpenAI."""
        response = self.openai_client.embeddings.create(
            input=query,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using OpenAI."""
        response = self.openai_client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        return [item.embedding for item in response.data]

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

    def retrieve_by_embedding(self, embedding: list[float], n_results: int = 3) -> list[dict]:
        """Retrieve chunks using a pre-computed embedding vector."""
        results = self.collection.query(
            query_embeddings=[embedding],
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
                temperature=0.0,
                timeout=15.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Помилка OpenAI API: {e}"

    def query(self, question: str, n_results: int = 3,
              strategy: str = 'vanilla', use_routing: bool = False,
              active_rag: str = None) -> dict:
        """
        Main query method with support for multiple RAG strategies.

        Args:
            question: The user's question in Ukrainian.
            n_results: Number of context chunks to retrieve.
            strategy: Query translation strategy. Options:
                - 'vanilla': Standard retrieval (default)
                - 'multi_query': Generate multiple query phrasings
                - 'rag_fusion': Multi-query + Reciprocal Rank Fusion
                - 'decompose': Break complex question into sub-questions
                - 'step_back': Abstract the question for broader retrieval
                - 'hyde': Hypothetical Document Embedding
            use_routing: Whether to apply query routing before retrieval.
            active_rag: Active RAG mode. Options:
                - None: Disabled (default)
                - 'crag': Corrective RAG
                - 'self_rag': Self-RAG
                - 'adaptive': Adaptive RAG

        Returns:
            dict with 'question', 'answer', 'context_chunks', 'n_chunks_used',
            'strategy', and optionally 'active_rag'.
        """

        if active_rag:
            if active_rag == 'crag':
                from src.active_rag.crag import run_crag
                return run_crag(self, question)
            elif active_rag == 'self_rag':
                from src.active_rag.self_rag import run_self_rag
                return run_self_rag(self, question)
            elif active_rag == 'adaptive':
                from src.active_rag.adaptive_rag import run_adaptive_rag
                return run_adaptive_rag(self, question)
            else:
                raise ValueError(f"Unknown active_rag mode: {active_rag}")

        if use_routing:
            from src.routing import QueryRouter
            router = QueryRouter(self.openai_client, self)
            routed = router.route(question)
            if routed:
                return routed

        if strategy == 'vanilla':
            context_chunks = self.retrieve(question, n_results=n_results)
        else:
            from src.query_translation import QueryTranslator
            translator = QueryTranslator(self.openai_client, self)
            context_chunks = translator.translate_and_retrieve(
                question, strategy=strategy, n_results=n_results
            )

        answer = self.generate(question, context_chunks)
        return {
            'question': question,
            'answer': answer,
            'context_chunks': context_chunks,
            'n_chunks_used': len(context_chunks),
            'strategy': strategy
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
