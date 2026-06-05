"""
Adaptive RAG for Ukrainian RAG Pipeline.

From RAG-from-scratch Notebooks 17-18:

Adaptive RAG dynamically selects the most appropriate RAG strategy
based on the complexity of the incoming question. It classifies each
query into one of three categories and routes accordingly:

    - SIMPLE: Answerable from general knowledge → no retrieval needed
    - STANDARD: Needs single-pass retrieval → vanilla RAG
    - COMPLEX: Needs multi-step reasoning → iterative RAG with
      query decomposition and grading

WHY LANGGRAPH?
    Adaptive RAG is implemented using LangGraph because it has genuine
    multi-path routing — the query classifier determines which of THREE
    different execution paths to take, each with different node sequences:

        classify → [complexity?]
                    ├── simple  → direct_generate
                    ├── standard → retrieve → generate
                    └── complex  → decompose → retrieve_loop → grade → generate

    This is a natural fit for a state graph with conditional edges.
    Unlike Self-RAG (which is a linear retry loop), Adaptive RAG has
    fundamentally different paths through the system depending on the
    classification result — exactly what graphs are designed for.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()



class AdaptiveState(TypedDict):
    """State that flows through the Adaptive RAG graph."""
    question: str
    complexity: str
    sub_questions: list[str]
    documents: list[dict]
    generation: str
    strategy_info: dict



def _classify_complexity(state: AdaptiveState, pipeline) -> AdaptiveState:
    """
    Node: Classify the question complexity.

    - simple: general knowledge, no retrieval needed
    - standard: single-pass retrieval is sufficient
    - complex: requires decomposition and multi-step retrieval
    """
    question = state["question"]

    prompt = (
        "Ти — класифікатор складності запитань. Визнач рівень складності "
        "наступного питання:\n\n"
        f"Питання: {question}\n\n"
        "Категорії:\n"
        '- "simple": просте питання, на яке можна відповісти із загальних знань '
        "(наприклад: 'Скільки днів у тижні?')\n"
        '- "standard": питання, яке потребує пошуку конкретної інформації '
        "(наприклад: 'Коли народився Тарас Шевченко?')\n"
        '- "complex": складне питання, яке потребує аналізу кількох аспектів '
        "або порівняння (наприклад: 'Як Голодомор вплинув на демографію України?')\n\n"
        'Відповідай ТІЛЬКИ одним словом: "simple", "standard", або "complex".'
    )

    response = pipeline.openai_client.chat.completions.create(
        model=pipeline.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )

    complexity = response.choices[0].message.content.strip().lower().strip('"\'')

    if complexity not in ("simple", "standard", "complex"):
        complexity = "standard"

    return {**state, "complexity": complexity}


def _direct_generate(state: AdaptiveState, pipeline) -> AdaptiveState:
    """
    Node: Generate answer directly without retrieval (for simple questions).
    """
    question = state["question"]

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

    info = state.get("strategy_info", {})
    info["path"] = "simple → direct_generate"

    return {**state, "generation": answer, "strategy_info": info}


def _standard_retrieve(state: AdaptiveState, pipeline) -> AdaptiveState:
    """
    Node: Standard single-pass retrieval.
    """
    question = state["question"]
    documents = pipeline.retrieve(question, n_results=3)

    info = state.get("strategy_info", {})
    info["path"] = "standard → retrieve → generate"
    info["retrieved_count"] = len(documents)

    return {**state, "documents": documents, "strategy_info": info}


def _decompose_question(state: AdaptiveState, pipeline) -> AdaptiveState:
    """
    Node: Decompose a complex question into sub-questions.
    """
    question = state["question"]

    prompt = (
        "Ти — помічник для аналізу складних питань. Розклади наступне складне "
        "питання на 2-4 простіших підпитання, які допоможуть побудувати "
        "повну відповідь. Генеруй підпитання українською мовою, "
        "кожне на новому рядку.\n\n"
        f"Складне питання: {question}\n\n"
        "Підпитання (по одному на рядок, без нумерації):"
    )

    response = pipeline.openai_client.chat.completions.create(
        model=pipeline.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    lines = response.choices[0].message.content.strip().split("\n")
    sub_questions = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in lines if line.strip()
    ]

    info = state.get("strategy_info", {})
    info["sub_questions"] = sub_questions

    return {**state, "sub_questions": sub_questions, "strategy_info": info}


def _complex_retrieve(state: AdaptiveState, pipeline) -> AdaptiveState:
    """
    Node: Retrieve context for each sub-question and combine.
    """
    import hashlib

    sub_questions = state["sub_questions"]
    all_docs = []
    seen = set()

    for sq in sub_questions:
        docs = pipeline.retrieve(sq, n_results=3)
        for doc in docs:
            raw = f"{doc.get('source', '')}:{doc.get('chunk_index', '')}:{doc.get('text', '')[:100]}"
            doc_id = hashlib.md5(raw.encode('utf-8')).hexdigest()
            if doc_id not in seen:
                seen.add(doc_id)
                all_docs.append(doc)

    info = state.get("strategy_info", {})
    info["path"] = "complex → decompose → multi_retrieve → generate"
    info["retrieved_count"] = len(all_docs)
    info["sub_question_count"] = len(sub_questions)

    return {**state, "documents": all_docs, "strategy_info": info}


def _generate_answer(state: AdaptiveState, pipeline) -> AdaptiveState:
    """
    Node: Generate answer from retrieved context.
    """
    question = state["question"]
    documents = state["documents"]

    answer = pipeline.generate(question, documents)

    return {**state, "generation": answer}



def _route_by_complexity(state: AdaptiveState) -> str:
    """
    Conditional edge: route to the appropriate path based on complexity.
    """
    return state["complexity"]



def _build_adaptive_graph(pipeline):
    """
    Build the Adaptive RAG graph using LangGraph.

    Graph structure:
        classify → [complexity?]
                    ├── simple   → direct_generate
                    ├── standard → standard_retrieve → generate
                    └── complex  → decompose → complex_retrieve → generate
    """
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(AdaptiveState)

    workflow.add_node("classify", lambda state: _classify_complexity(state, pipeline))
    workflow.add_node("direct_generate", lambda state: _direct_generate(state, pipeline))
    workflow.add_node("standard_retrieve", lambda state: _standard_retrieve(state, pipeline))
    workflow.add_node("decompose", lambda state: _decompose_question(state, pipeline))
    workflow.add_node("complex_retrieve", lambda state: _complex_retrieve(state, pipeline))
    workflow.add_node("generate", lambda state: _generate_answer(state, pipeline))

    workflow.set_entry_point("classify")

    workflow.add_conditional_edges(
        "classify",
        _route_by_complexity,
        {
            "simple": "direct_generate",
            "standard": "standard_retrieve",
            "complex": "decompose",
        }
    )

    workflow.add_edge("direct_generate", END)

    workflow.add_edge("standard_retrieve", "generate")
    workflow.add_edge("generate", END)

    workflow.add_edge("decompose", "complex_retrieve")
    workflow.add_edge("complex_retrieve", "generate")

    return workflow.compile()



def run_adaptive_rag(pipeline, question: str) -> dict:
    """
    Run the Adaptive RAG pipeline on a question.

    Args:
        pipeline: The RAGPipeline instance.
        question: The user's question in Ukrainian.

    Returns:
        Standard pipeline result dict with additional Adaptive RAG metadata.
    """
    graph = _build_adaptive_graph(pipeline)

    initial_state: AdaptiveState = {
        "question": question,
        "complexity": "",
        "sub_questions": [],
        "documents": [],
        "generation": "",
        "strategy_info": {},
    }

    final_state = graph.invoke(initial_state)

    return {
        "question": question,
        "answer": final_state["generation"],
        "context_chunks": final_state["documents"],
        "n_chunks_used": len(final_state["documents"]),
        "strategy": "vanilla",
        "active_rag": "adaptive",
        "adaptive_info": {
            "complexity": final_state["complexity"],
            **final_state["strategy_info"],
        },
    }
