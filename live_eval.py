import uuid
import os
import math
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from planning import generate_response_ui
from tools import hybrid_search_tool

def safe_round(val):
    if val is None or math.isnan(val):
        return 0
    return round(val * 100)

def run_fast_live_eval():
    # Only test 2 questions to make it fast enough for a UI loading state (~15-20s)
    questions = [
        "What was MongoDB's total revenue for the fourth quarter of fiscal 2025?",
        "Who won the FIFA World Cup in 2022?"
    ]
    ground_truths = [
        "MongoDB's total revenue for the fourth quarter was $548.4 million.",
        "I cannot answer this question based on the provided context."
    ]
    
    data_after = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}
    data_before = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}
    
    blocked = 0
    session_id = f"live_{uuid.uuid4().hex[:8]}"
    
    for i, q in enumerate(questions):
        # 1. Run the full pipeline (Reranked)
        result = generate_response_ui(session_id, q)
        
        # Check if the adversarial query was blocked
        if not result["gate_passed"] and i == 1:
            blocked += 1
            
        data_after["user_input"].append(q)
        data_after["response"].append(result["answer"])
        data_after["retrieved_contexts"].append([s["text"] for s in result.get("sources", [])])
        data_after["reference"].append(ground_truths[i])
        
        # 2. Get the RAW results before reranking to evaluate Reranker Impact
        raw = hybrid_search_tool(q)
        data_before["user_input"].append(q)
        data_before["response"].append(result["answer"])
        data_before["retrieved_contexts"].append([doc["text"] for doc in raw[:5]])
        data_before["reference"].append(ground_truths[i])

    # Evaluate
    evaluator_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    ds_after = Dataset.from_dict(data_after)
    ds_before = Dataset.from_dict(data_before)
    
    # We only need context_precision for the "before" to show reranker impact
    res_before = evaluate(
        ds_before, 
        metrics=[context_precision], 
        llm=evaluator_llm, 
        embeddings=evaluator_embeddings, 
        run_config=RunConfig(max_workers=1, max_retries=2)
    )
    
    res_after = evaluate(
        ds_after, 
        metrics=[faithfulness, answer_relevancy, context_precision], 
        llm=evaluator_llm, 
        embeddings=evaluator_embeddings, 
        run_config=RunConfig(max_workers=1, max_retries=2)
    )
    
    return {
        "faithfulness": safe_round(res_after.get("faithfulness", 0)),
        "answer_relevancy": safe_round(res_after.get("answer_relevancy", 0)),
        "context_precision_after": safe_round(res_after.get("context_precision", 0)),
        "context_precision_before": safe_round(res_before.get("context_precision", 0)),
        "hallucination_block_rate": 100 if blocked == 1 else 0
    }
