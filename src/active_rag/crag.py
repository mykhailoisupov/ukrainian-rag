"""
Corrective RAG (CRAG) for Ukrainian RAG Pipeline.

From RAG-from-scratch Notebook 15:

CRAG adds a self-correction layer to standard RAG. After retrieving
documents, an LLM *grades* each document for relevance. Based on the
grading outcome:

    - ALL relevant → proceed to generation with retrieved context
    - ANY irrelevant → supplement with web search results (via Tavily API)
    - ALL irrelevant → discard retrieved docs entirely, rely on web search

WHY LANGGRAPH?
    CRAG is implemented using LangGraph because the decision flow is
    naturally a directed graph with conditional edges:

        retrieve → grade_documents → [relevant?]
                                        ├── yes → generate
                                        └── no  → web_search → generate

    LangGraph makes these conditional transitions explicit and declarative,
    which is cleaner than nested if/else chains. It also makes the flow
    easy to visualize, extend (e.g. add a "retry with rephrased query"
    node), and debug via state inspection at each node.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()



class CRAGState(TypedDict):
    """State that flows through the CRAG graph."""
    question: str
    documents: list[dict]
    web_results: list[dict]
    relevance_scores: list[str]
    generation: str
    strategy_info: dict



def _retrieve(state: CRAGState, pipeline) -> CRAGState:
    """Node: Retrieve documents from the vector store."""
    question = state["question"]
    documents = pipeline.retrieve(question, n_results=4)
    return {**state, "documents": documents}


def _grade_documents(state: CRAGState, pipeline) -> CRAGState:
    """
    Node: Grade each retrieved document for relevance to the question.
    Uses the LLM as a binary classifier: "relevant" or "irrelevant".
    """
    question = state["question"]
    documents = state["documents"]
    scores = []

    for doc in documents:
        prompt = (
            "Ти — оцінювач релевантності. Дано питання та отриманий документ. "
            "Визнач, чи містить документ інформацію, корисну для відповіді "
            "на питання.\n\n"
            f"Питання: {question}\n\n"
            f"Документ: {doc['text']}\n\n"
            "Відповідай ТІЛЬКИ одним словом: 'relevant' або 'irrelevant'."
        )

        response = pipeline.openai_client.chat.completions.create(
            model=pipeline.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        grade = response.choices[0].message.content.strip().lower()
        scores.append("relevant" if "relevant" in grade and "irrelevant" not in grade else "irrelevant")

    return {**state, "relevance_scores": scores}


def _web_search(state: CRAGState, pipeline) -> CRAGState:
    """
    Node: Perform web search to supplement or replace retrieved context.
    Uses the Tavily API for search.
    """
    question = state["question"]

    try:
        from tavily import TavilyClient
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        if not tavily_key:
            return {**state, "web_results": [{"text": "(Web search skipped — TAVILY_API_KEY not set)", "source": "web", "chunk_index": 0, "distance": 0.0}]}

        tavily = TavilyClient(api_key=tavily_key)
        search_results = tavily.search(
            query=question,
            search_depth="advanced",
            max_results=3,
        )

        web_docs = []
        for i, result in enumerate(search_results.get("results", [])):
            web_docs.append({
                "text": result.get("content", ""),
                "source": result.get("url", "web"),
                "chunk_index": i,
                "distance": 0.0,
            })

        return {**state, "web_results": web_docs}

    except ImportError:
        return {**state, "web_results": [{"text": "(Web search skipped — tavily not installed)", "source": "web", "chunk_index": 0, "distance": 0.0}]}
    except Exception as e:
        return {**state, "web_results": [{"text": f"(Web search error: {e})", "source": "web", "chunk_index": 0, "distance": 0.0}]}


def _generate(state: CRAGState, pipeline) -> CRAGState:
    """
    Node: Generate answer using the best available context.
    Uses relevant retrieved docs + web results (if any).
    """
    question = state["question"]
    documents = state["documents"]
    scores = state["relevance_scores"]
    web_results = state.get("web_results", [])

    relevant_docs = [
        doc for doc, score in zip(documents, scores)
        if score == "relevant"
    ]

    all_context = relevant_docs + web_results

    if not all_context:
        all_context = documents

    answer = pipeline.generate(question, all_context)

    strategy_info = {
        "total_retrieved": len(documents),
        "relevant_count": len(relevant_docs),
        "irrelevant_count": len(documents) - len(relevant_docs),
        "web_search_triggered": len(web_results) > 0,
        "web_results_count": len(web_results),
    }

    return {**state, "generation": answer, "strategy_info": strategy_info}



def _should_web_search(state: CRAGState) -> str:
    """
    Conditional edge: decide whether to trigger web search.
    Returns "web_search" or "generate".
    """
    scores = state["relevance_scores"]
    irrelevant_count = scores.count("irrelevant")

    if irrelevant_count > 0:
        return "web_search"
    else:
        return "generate"



def _build_crag_graph(pipeline):
    """
    Build the CRAG graph using LangGraph.

    Graph structure:
        retrieve → grade_documents → [any irrelevant?]
                                        ├── yes → web_search → generate
                                        └── no  → generate
    """
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(CRAGState)

    workflow.add_node("retrieve", lambda state: _retrieve(state, pipeline))
    workflow.add_node("grade_documents", lambda state: _grade_documents(state, pipeline))
    workflow.add_node("web_search", lambda state: _web_search(state, pipeline))
    workflow.add_node("generate", lambda state: _generate(state, pipeline))

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        _should_web_search,
        {
            "web_search": "web_search",
            "generate": "generate",
        }
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()



def run_crag(pipeline, question: str) -> dict:
    """
    Run the CRAG pipeline on a question.

    Args:
        pipeline: The RAGPipeline instance.
        question: The user's question in Ukrainian.

    Returns:
        Standard pipeline result dict with additional CRAG metadata.
    """
    graph = _build_crag_graph(pipeline)

    initial_state: CRAGState = {
        "question": question,
        "documents": [],
        "web_results": [],
        "relevance_scores": [],
        "generation": "",
        "strategy_info": {},
    }

    final_state = graph.invoke(initial_state)

    relevant_docs = [
        doc for doc, score in zip(final_state["documents"], final_state["relevance_scores"])
        if score == "relevant"
    ]
    all_context = relevant_docs + final_state.get("web_results", [])

    return {
        "question": question,
        "answer": final_state["generation"],
        "context_chunks": all_context,
        "n_chunks_used": len(all_context),
        "strategy": "vanilla",
        "active_rag": "crag",
        "crag_info": final_state["strategy_info"],
    }
