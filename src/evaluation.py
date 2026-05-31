import os
import sys
import json
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

def run_evaluation():
    pipeline = RAGPipeline()
    
    questions = []
    answers = []
    ground_truths = []
    contexts = []
    
    print("Running queries through RAG pipeline...")
    for item in TEST_QUESTIONS:
        q = item['question']
        gt = item['ground_truth']
        
        print(f"Querying: {q}")
        res = pipeline.query(q)
        
        questions.append(q)
        answers.append(res['answer'])
        ground_truths.append(gt)
        contexts.append([chunk['text'] for chunk in res['context_chunks']])
        
    dataset = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    })
    
    print("Initializing RAGAS evaluator models...")
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("Running RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    scores = dict(results)
    
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
        
    df = pd.DataFrame([scores])
    print("\nEvaluation Metrics Summary:")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    return scores

if __name__ == '__main__':
    run_evaluation()
    print("Evaluation complete. Results saved to data/evaluation_results.json")
