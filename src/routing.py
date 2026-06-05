"""
Query Routing for Ukrainian RAG Pipeline.

Implements 2 routing strategies from RAG-from-scratch Notebooks 10-11:

1. Logical Routing — LLM classifies the query and picks the best data source
   or prompt template based on structured reasoning.

2. Semantic Routing — Embed the query and pre-defined prompt descriptions,
   then route to the prompt with highest cosine similarity.

In the context of the Ukrainian RAG system, routing decides which *prompt
template* or *retrieval strategy* to use based on the nature of the question.
"""

import json
import numpy as np



PROMPT_TEMPLATES = {
    "factual": {
        "description": "Фактичні питання про конкретні дати, імена, місця, числа",
        "system_prompt": (
            "Ти корисний асистент, який відповідає на фактичні питання "
            "виключно на основі наданого контексту українською мовою. "
            "Давай точні, конкретні відповіді з числами та датами. "
            "Якщо відповідь не міститься в контексті, скажи: "
            "'Я не можу знайти цю інформацію в наданих документах.'"
        ),
    },
    "analytical": {
        "description": "Аналітичні питання, що вимагають пояснення, порівняння, причинно-наслідкових зв'язків",
        "system_prompt": (
            "Ти корисний асистент, який надає глибокий аналіз та пояснення "
            "на основі наданого контексту українською мовою. Структуруй "
            "відповідь логічно, вказуй на причинно-наслідкові зв'язки "
            "та наводь аргументи з контексту. "
            "Якщо відповідь не міститься в контексті, скажи: "
            "'Я не можу знайти цю інформацію в наданих документах.'"
        ),
    },
    "summary": {
        "description": "Питання, що вимагають узагальнення, огляду або стислого викладу теми",
        "system_prompt": (
            "Ти корисний асистент, який створює стислі та інформативні "
            "узагальнення на основі наданого контексту українською мовою. "
            "Виділяй ключові тези та структуруй інформацію послідовно. "
            "Якщо відповідь не міститься в контексті, скажи: "
            "'Я не можу знайти цю інформацію в наданих документах.'"
        ),
    },
}


class QueryRouter:
    """
    Routes queries to the optimal prompt template or retrieval strategy.

    Args:
        openai_client: An initialized OpenAI client instance.
        pipeline: The RAGPipeline instance (for access to retrieve/embed/generate).
    """

    def __init__(self, openai_client, pipeline):
        self.client = openai_client
        self.pipeline = pipeline
        self.model = pipeline.openai_model
        self._prompt_embeddings = None


    def route(self, question: str, method: str = 'logical') -> dict | None:
        """
        Route the question and execute the full pipeline with the
        appropriate prompt template.

        Args:
            question: The user's question.
            method: 'logical' or 'semantic'.

        Returns:
            A full pipeline result dict, or None if routing is skipped.
        """
        if method == 'logical':
            template_key = self.logical_route(question)
        elif method == 'semantic':
            template_key = self.semantic_route(question)
        else:
            raise ValueError(f"Unknown routing method: {method}")

        context_chunks = self.pipeline.retrieve(question, n_results=3)

        template = PROMPT_TEMPLATES[template_key]
        answer = self._generate_with_template(question, context_chunks, template)

        return {
            'question': question,
            'answer': answer,
            'context_chunks': context_chunks,
            'n_chunks_used': len(context_chunks),
            'strategy': 'vanilla',
            'routing': {
                'method': method,
                'selected_template': template_key,
            },
        }


    def logical_route(self, question: str) -> str:
        """
        Use the LLM to classify the question type and select the best
        prompt template via structured reasoning.
        """
        template_descriptions = "\n".join(
            f'- "{key}": {tmpl["description"]}'
            for key, tmpl in PROMPT_TEMPLATES.items()
        )

        prompt = (
            "Ти — класифікатор запитів. Дано питання користувача та список "
            "доступних шаблонів відповіді. Обери ОДИН найкращий шаблон.\n\n"
            f"Доступні шаблони:\n{template_descriptions}\n\n"
            f"Питання: {question}\n\n"
            'Відповідай ТІЛЬКИ назвою шаблону (одне слово): '
            '"factual", "analytical", або "summary".'
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        chosen = response.choices[0].message.content.strip().lower().strip('"\'')

        if chosen not in PROMPT_TEMPLATES:
            chosen = "factual"

        return chosen


    def semantic_route(self, question: str) -> str:
        """
        Embed the question and each prompt template description, then
        route to the template whose description is most semantically
        similar to the question.
        """
        if self._prompt_embeddings is None:
            self._prompt_embeddings = self._compute_prompt_embeddings()

        question_embedding = np.array(self.pipeline.embed_query(question))

        best_key = None
        best_score = -1.0

        for key, emb in self._prompt_embeddings.items():
            score = _cosine_similarity(question_embedding, emb)
            if score > best_score:
                best_score = score
                best_key = key

        return best_key

    def _compute_prompt_embeddings(self) -> dict[str, np.ndarray]:
        """Embed all prompt template descriptions."""
        descriptions = [tmpl["description"] for tmpl in PROMPT_TEMPLATES.values()]
        embeddings = self.pipeline.embed_texts(descriptions)
        return {
            key: np.array(emb)
            for key, emb in zip(PROMPT_TEMPLATES.keys(), embeddings)
        }


    def _generate_with_template(self, query: str, context_chunks: list[dict],
                                 template: dict) -> str:
        """Generate an answer using a specific prompt template."""
        context_str = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_str += f"[{i}] (Джерело: {chunk['source']}): {chunk['text']}\n\n"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": template["system_prompt"]},
                    {
                        "role": "user",
                        "content": f"Контекст:\n{context_str}\n\nПитання: {query}",
                    },
                ],
                temperature=0.0,
                timeout=15.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Помилка OpenAI API: {e}"



def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)
