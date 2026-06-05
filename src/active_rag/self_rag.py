"""
Self-RAG for Ukrainian RAG Pipeline.

From RAG-from-scratch Notebook 16:

Self-RAG adds multiple layers of self-reflection to the RAG process.
The LLM itself decides at each step whether to proceed or loop:

    1. DECIDE whether retrieval is even needed for this question
    2. If retrieving: GRADE each document for relevance
    3. GENERATE an answer from the relevant context
    4. CHECK if the answer is grounded in context (hallucination check)
    5. CHECK if the answer actually addresses the question
    6. If any check fails: retry with adjusted parameters

WHY HAND-ROLLED (not LangGraph)?
    Self-RAG is implemented as a simple procedural loop rather than a
    LangGraph because:

    - The flow is essentially linear with retry logic (not a complex graph)
    - There are no parallel branches or multi-path routing
    - A while loop with early returns is more readable than a graph
      for this pattern
    - Debugging is trivial: just add print statements to the loop body
    - It's easier to understand *exactly* what's happening step by step

    Compare this to CRAG, which has a genuine branching decision
    (web search vs. direct generation) that maps naturally to a graph.
    Self-RAG is just: try → check → retry if needed.
"""

from dotenv import load_dotenv

load_dotenv()



def _needs_retrieval(question: str, pipeline) -> bool:
    """
    Step 1: Ask the LLM whether retrieval is needed.
    Simple factual questions like "What is 2+2?" don't need retrieval.
    """
    prompt = (
        "Ти — класифікатор запитів. Визнач, чи потрібно шукати інформацію "
        "у зовнішній базі даних для відповіді на це питання, чи можна "
        "відповісти із загальних знань.\n\n"
        f"Питання: {question}\n\n"
        "Відповідай ТІЛЬКИ одним словом: 'yes' (потрібен пошук) або 'no' (не потрібен)."
    )

    response = pipeline.openai_client.chat.completions.create(
        model=pipeline.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )

    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer


def _grade_relevance(question: str, doc_text: str, pipeline) -> bool:
    """Step 2: Grade a single document for relevance to the question."""
    prompt = (
        "Ти — оцінювач релевантності документів. Визнач, чи містить "
        "наданий документ інформацію, корисну для відповіді на питання.\n\n"
        f"Питання: {question}\n\n"
        f"Документ: {doc_text}\n\n"
        "Відповідай ТІЛЬКИ одним словом: 'yes' (релевантний) або 'no' (нерелевантний)."
    )

    response = pipeline.openai_client.chat.completions.create(
        model=pipeline.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )

    return "yes" in response.choices[0].message.content.strip().lower()


def _check_hallucination(answer: str, context_chunks: list[dict], pipeline) -> bool:
    """
    Step 4: Check whether the generated answer is grounded in the context.
    Returns True if the answer IS grounded (no hallucination).
    """
    context_str = "\n\n".join(chunk["text"] for chunk in context_chunks)

    prompt = (
        "Ти — оцінювач якості відповідей. Визнач, чи базується надана "
        "відповідь виключно на фактах із наданого контексту. "
        "Чи є в відповіді інформація, якої НЕМАЄ в контексті?\n\n"
        f"Контекст:\n{context_str}\n\n"
        f"Відповідь: {answer}\n\n"
        "Відповідай ТІЛЬКИ одним словом: 'grounded' (базується на контексті) "
        "або 'hallucination' (містить вигадану інформацію)."
    )

    response = pipeline.openai_client.chat.completions.create(
        model=pipeline.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=15,
    )

    return "grounded" in response.choices[0].message.content.strip().lower()


def _check_usefulness(question: str, answer: str, pipeline) -> bool:
    """
    Step 5: Check whether the answer actually addresses the question.
    Returns True if the answer IS useful.
    """
    prompt = (
        "Ти — оцінювач корисності відповідей. Визнач, чи відповідає "
        "надана відповідь на поставлене питання повністю та по суті.\n\n"
        f"Питання: {question}\n\n"
        f"Відповідь: {answer}\n\n"
        "Відповідай ТІЛЬКИ одним словом: 'useful' (корисна) або 'not_useful' (некорисна)."
    )

    response = pipeline.openai_client.chat.completions.create(
        model=pipeline.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=15,
    )

    return "useful" in response.choices[0].message.content.strip().lower()



def run_self_rag(pipeline, question: str, max_retries: int = 2) -> dict:
    """
    Run the Self-RAG pipeline on a question.

    This is a simple procedural loop (not a graph) because the flow is
    linear: retrieve → grade → generate → check → retry if needed.

    Args:
        pipeline: The RAGPipeline instance.
        question: The user's question in Ukrainian.
        max_retries: Maximum number of retry attempts.

    Returns:
        Standard pipeline result dict with additional Self-RAG metadata.
    """
    self_rag_info = {
        "retrieval_needed": True,
        "attempts": 0,
        "relevant_docs_per_attempt": [],
        "hallucination_checks": [],
        "usefulness_checks": [],
    }

    needs_retrieval = _needs_retrieval(question, pipeline)
    self_rag_info["retrieval_needed"] = needs_retrieval

    if not needs_retrieval:
        try:
            response = pipeline.openai_client.chat.completions.create(
                model=pipeline.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ти корисний асистент. Відповідай на питання "
                            "українською мовою, коротко та по суті."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Помилка OpenAI API: {e}"

        return {
            "question": question,
            "answer": answer,
            "context_chunks": [],
            "n_chunks_used": 0,
            "strategy": "vanilla",
            "active_rag": "self_rag",
            "self_rag_info": self_rag_info,
        }

    n_results = 4
    best_answer = None
    best_context = []

    for attempt in range(max_retries + 1):
        self_rag_info["attempts"] = attempt + 1

        all_docs = pipeline.retrieve(question, n_results=n_results)
        relevant_docs = [
            doc for doc in all_docs
            if _grade_relevance(question, doc["text"], pipeline)
        ]
        self_rag_info["relevant_docs_per_attempt"].append(len(relevant_docs))

        context = relevant_docs if relevant_docs else all_docs

        answer = pipeline.generate(question, context)

        is_grounded = _check_hallucination(answer, context, pipeline)
        self_rag_info["hallucination_checks"].append(is_grounded)

        is_useful = _check_usefulness(question, answer, pipeline)
        self_rag_info["usefulness_checks"].append(is_useful)

        if is_grounded and is_useful:
            return {
                "question": question,
                "answer": answer,
                "context_chunks": context,
                "n_chunks_used": len(context),
                "strategy": "vanilla",
                "active_rag": "self_rag",
                "self_rag_info": self_rag_info,
            }

        if best_answer is None or (is_grounded and not is_useful):
            best_answer = answer
            best_context = context

        n_results += 2

    return {
        "question": question,
        "answer": best_answer or answer,
        "context_chunks": best_context or context,
        "n_chunks_used": len(best_context or context),
        "strategy": "vanilla",
        "active_rag": "self_rag",
        "self_rag_info": self_rag_info,
    }
