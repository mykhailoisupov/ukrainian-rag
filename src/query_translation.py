"""
Query Translation Strategies for Ukrainian RAG Pipeline.

Implements 5 query translation techniques from the RAG-from-scratch series
(Notebooks 5-9):

1. Multi-Query: Generate multiple rephrasings → retrieve for each → union
2. RAG-Fusion: Multi-Query + Reciprocal Rank Fusion re-ranking
3. Decomposition: Break complex question into sub-questions
4. Step-back: Abstract the question for broader context retrieval
5. HyDE: Generate a hypothetical answer, embed it, retrieve by similarity

All strategies are designed for Ukrainian-language queries.
"""

import hashlib


class QueryTranslator:
    """
    Translates a user query into alternative forms to improve retrieval quality.

    Args:
        openai_client: An initialized OpenAI client instance.
        pipeline: The RAGPipeline instance (for access to retrieve/embed methods).
    """

    def __init__(self, openai_client, pipeline):
        self.client = openai_client
        self.pipeline = pipeline
        self.model = pipeline.openai_model


    def translate_and_retrieve(self, question: str, strategy: str,
                               n_results: int = 3) -> list[dict]:
        """
        Apply the specified query translation strategy and return
        de-duplicated context chunks.
        """
        strategy_map = {
            'multi_query': self._multi_query,
            'rag_fusion': self._rag_fusion,
            'decompose': self._decompose,
            'step_back': self._step_back,
            'hyde': self._hyde,
        }

        if strategy not in strategy_map:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(strategy_map.keys())}"
            )

        return strategy_map[strategy](question, n_results)


    def _generate_alternative_queries(self, question: str, n: int = 3) -> list[str]:
        """Use the LLM to generate N alternative phrasings of the question."""
        prompt = (
            f"Ти — помічник зі штучного інтелекту. Твоє завдання — згенерувати "
            f"{n} різних версій заданого питання користувача, щоб отримати "
            f"релевантні документи з векторної бази даних. Генеруй альтернативні "
            f"питання українською мовою, кожне на новому рядку.\n\n"
            f"Оригінальне питання: {question}\n\n"
            f"Альтернативні питання (по одному на рядок, без нумерації):"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        lines = response.choices[0].message.content.strip().split("\n")
        alternatives = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in lines if line.strip()
        ]
        return alternatives[:n]

    def _multi_query(self, question: str, n_results: int) -> list[dict]:
        """
        Generate alternative query phrasings, retrieve for each, and
        return the union of all results (de-duplicated).
        """
        alt_queries = self._generate_alternative_queries(question)
        all_queries = [question] + alt_queries

        all_chunks = []
        seen = set()

        for q in all_queries:
            chunks = self.pipeline.retrieve(q, n_results=n_results)
            for chunk in chunks:
                chunk_id = _chunk_hash(chunk)
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    all_chunks.append(chunk)

        return all_chunks


    def _rag_fusion(self, question: str, n_results: int) -> list[dict]:
        """
        Multi-Query + Reciprocal Rank Fusion (RRF).

        For each alternative query, retrieve ranked results. Then combine
        all result lists using RRF to produce a single re-ranked list.
        """
        alt_queries = self._generate_alternative_queries(question)
        all_queries = [question] + alt_queries

        ranked_lists: list[list[dict]] = []
        for q in all_queries:
            chunks = self.pipeline.retrieve(q, n_results=n_results)
            ranked_lists.append(chunks)

        fused = _reciprocal_rank_fusion(ranked_lists)

        return fused[:n_results]


    def _decompose(self, question: str, n_results: int) -> list[dict]:
        """
        Decompose a complex question into simpler sub-questions,
        retrieve context for each, and combine.
        """
        prompt = (
            "Ти — помічник зі штучного інтелекту. Розклади наступне складне "
            "питання на 2-4 простіших підпитання, які допоможуть відповісти на "
            "оригінальне питання. Генеруй підпитання українською мовою, "
            "кожне на новому рядку.\n\n"
            f"Питання: {question}\n\n"
            "Підпитання (по одному на рядок, без нумерації):"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        lines = response.choices[0].message.content.strip().split("\n")
        sub_questions = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in lines if line.strip()
        ]

        all_chunks = []
        seen = set()

        for sq in sub_questions:
            chunks = self.pipeline.retrieve(sq, n_results=n_results)
            for chunk in chunks:
                chunk_id = _chunk_hash(chunk)
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    all_chunks.append(chunk)

        return all_chunks


    def _step_back(self, question: str, n_results: int) -> list[dict]:
        """
        Generate a more abstract 'step-back' version of the question
        to retrieve broader context, then combine with original results.
        """
        prompt = (
            "Ти — помічник зі штучного інтелекту. Дано конкретне питання. "
            "Сформулюй більш загальне, абстрактне питання (step-back question), "
            "яке допоможе отримати ширший контекст для відповіді на оригінальне "
            "питання. Відповідай українською мовою.\n\n"
            f"Оригінальне питання: {question}\n\n"
            "Більш загальне питання:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        step_back_question = response.choices[0].message.content.strip()

        original_chunks = self.pipeline.retrieve(question, n_results=n_results)
        stepback_chunks = self.pipeline.retrieve(step_back_question, n_results=n_results)

        all_chunks = []
        seen = set()
        for chunk in original_chunks + stepback_chunks:
            chunk_id = _chunk_hash(chunk)
            if chunk_id not in seen:
                seen.add(chunk_id)
                all_chunks.append(chunk)

        return all_chunks


    def _hyde(self, question: str, n_results: int) -> list[dict]:
        """
        Generate a hypothetical answer document, embed it with OpenAI,
        and use that embedding to retrieve real documents.

        The insight: embedding a *plausible* answer is often closer in
        vector space to the real answer chunks than the question itself.
        """
        prompt = (
            "Ти — помічник зі штучного інтелекту. Напиши короткий, детальний "
            "параграф, який міг би бути відповіддю на наступне питання. "
            "Відповідай українською мовою, навіть якщо ти не впевнений — "
            "просто згенеруй правдоподібну відповідь.\n\n"
            f"Питання: {question}\n\n"
            "Гіпотетична відповідь:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        hypothetical_doc = response.choices[0].message.content.strip()

        hyde_embedding = self.pipeline.embed_query(hypothetical_doc)
        return self.pipeline.retrieve_by_embedding(hyde_embedding, n_results=n_results)



def _chunk_hash(chunk: dict) -> str:
    """Create a stable hash for a chunk to detect duplicates."""
    raw = f"{chunk.get('source', '')}:{chunk.get('chunk_index', '')}:{chunk.get('text', '')[:100]}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) — combines multiple ranked lists into one.

    For each document, compute:  score = Σ  1 / (k + rank_i)
    across all lists where the document appears.

    Args:
        ranked_lists: A list of ranked result lists.
        k: Smoothing constant (default 60, from the original RRF paper).

    Returns:
        A single list of chunks sorted by RRF score (descending).
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            chunk_id = _chunk_hash(chunk)
            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [chunk_lookup[cid] for cid in sorted_ids]
