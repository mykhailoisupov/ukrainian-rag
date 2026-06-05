import os
import sys
import json
import argparse
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.rag_pipeline import RAGPipeline

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

TEST_QUESTIONS = [
    {
        "question": "Коли народився Тарас Шевченко?",
        "ground_truth": "Тарас Шевченко народився 9 березня 1814 року."
    },
    {
        "question": "Що таке штучний інтелект?",
        "ground_truth": "Штучний інтелект (ШІ) — це здатність обчислювальних систем виконувати завдання, які зазвичай співмірні з можливостями людського інтелекту, такими як навчання, міркування, розв'язання питань, сприйняття та ухвалення рішень."
    },
    {
        "question": "Коли сталася аварія на Чорнобильській АЕС?",
        "ground_truth": "Аварія на Чорнобильській АЕС сталася 26 квітня 1986 року."
    },
    {
        "question": "Яке місто є столицею та найбільшим містом України?",
        "ground_truth": "Столицею та найбільшим містом України є Київ."
    },
    {
        "question": "В які роки відбувся Голодомор в Україні?",
        "ground_truth": "Голодомор в Україні відбувся у 1932–1933 роках."
    },
    {
        "question": "Який орган є єдиним законодавчим органом в Україні?",
        "ground_truth": "Єдиним законодавчим органом (парламентом) України є Верховна Рада України."
    },
    {
        "question": "У якому селі народився Тарас Шевченко?",
        "ground_truth": "Тарас Шевченко народився в селі Моринці Київської губернії."
    },
    {
        "question": "Яке місто є найбільшим на заході України?",
        "ground_truth": "Найбільшим містом на заході України є Львів."
    },
    {
        "question": "В який період було збудовано першу чергу ЧАЕС?",
        "ground_truth": "Першу чергу Чорнобильської АЕС та місто Прип'ять було побудовано у 1970—1977 роках."
    },
    {
        "question": "Чим займається машинне навчання?",
        "ground_truth": "Машинне навчання є галуззю інформатики, що вивчає методи та алгоритми, які дають комп'ютерам можливість навчатися без безпосереднього програмування."
    }
]

AVAILABLE_STRATEGIES = [
    'vanilla', 'multi_query', 'rag_fusion', 'decompose', 'step_back', 'hyde'
]

AVAILABLE_ACTIVE_RAG = ['crag', 'self_rag', 'adaptive']


def run_evaluation(strategies: list[str] = None, active_rag_modes: list[str] = None):
    """
    Run RAGAS evaluation across one or more strategies.

    Args:
        strategies: List of query translation strategies to evaluate.
                    Defaults to ['vanilla'] if not specified.
        active_rag_modes: List of active RAG modes to evaluate.
                          Defaults to None (disabled).

    Returns:
        dict mapping strategy name → RAGAS scores.
    """
    if strategies is None:
        strategies = ['vanilla']

    pipeline = RAGPipeline()
    all_results = {}

    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"Evaluating strategy: {strategy}")
        print(f"{'=' * 60}")

        questions = []
        answers = []
        ground_truths = []
        contexts = []

        for item in TEST_QUESTIONS:
            q = item['question']
            gt = item['ground_truth']

            print(f"  Querying: {q}")
            try:
                res = pipeline.query(q, strategy=strategy)
                questions.append(q)
                answers.append(res['answer'])
                ground_truths.append(gt)
                contexts.append([chunk['text'] for chunk in res['context_chunks']])
            except Exception as e:
                print(f"  ERROR: {e}")
                questions.append(q)
                answers.append(f"Error: {e}")
                ground_truths.append(gt)
                contexts.append([""])

        scores = _compute_ragas_scores(questions, answers, ground_truths, contexts)
        all_results[strategy] = scores
        _print_scores(strategy, scores)

    if active_rag_modes:
        for mode in active_rag_modes:
            print(f"\n{'=' * 60}")
            print(f"Evaluating active RAG: {mode}")
            print(f"{'=' * 60}")

            questions = []
            answers = []
            ground_truths = []
            contexts = []

            for item in TEST_QUESTIONS:
                q = item['question']
                gt = item['ground_truth']

                print(f"  Querying: {q}")
                try:
                    res = pipeline.query(q, active_rag=mode)
                    questions.append(q)
                    answers.append(res['answer'])
                    ground_truths.append(gt)
                    contexts.append([chunk['text'] for chunk in res['context_chunks']])
                except Exception as e:
                    print(f"  ERROR: {e}")
                    questions.append(q)
                    answers.append(f"Error: {e}")
                    ground_truths.append(gt)
                    contexts.append([""])

            scores = _compute_ragas_scores(questions, answers, ground_truths, contexts)
            all_results[f"active_{mode}"] = scores
            _print_scores(f"active_{mode}", scores)

    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    if len(all_results) > 1:
        _print_comparison_table(all_results)

    return all_results


def _compute_ragas_scores(questions, answers, ground_truths, contexts):
    """Compute RAGAS scores for a set of question-answer pairs."""
    dataset = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    })

    print("  Running RAGAS evaluation...")
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    return dict(results)


def _print_scores(strategy_name, scores):
    """Print scores for a single strategy."""
    print(f"\n  Results for '{strategy_name}':")
    for metric, value in scores.items():
        print(f"    {metric}: {value:.4f}")


def _print_comparison_table(all_results):
    """Print a comparison table across all evaluated strategies."""
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE")
    print(f"{'=' * 80}")

    df = pd.DataFrame(all_results).T
    df.index.name = 'Strategy'
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"{'=' * 80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RAGAS evaluation on the Ukrainian RAG pipeline.')
    parser.add_argument(
        '--strategies', nargs='+', default=['vanilla'],
        choices=AVAILABLE_STRATEGIES,
        help='Query translation strategies to evaluate.'
    )
    parser.add_argument(
        '--active-rag', nargs='+', default=None,
        choices=AVAILABLE_ACTIVE_RAG,
        help='Active RAG modes to evaluate.'
    )
    args = parser.parse_args()

    run_evaluation(strategies=args.strategies, active_rag_modes=args.active_rag)
    print("\nEvaluation complete. Results saved to data/evaluation_results.json")
