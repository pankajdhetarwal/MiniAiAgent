import sys
import types
import os
import time

# Monkey-patch to fix RAGAS 0.3.1 compatibility issue with Langchain Community
sys.modules['langchain_community.chat_models.vertexai'] = types.ModuleType('langchain_community.chat_models.vertexai')
sys.modules['langchain_community.chat_models.vertexai'].ChatVertexAI = None

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Import your agent's pipeline
from planning import generate_response
from tools import hybrid_search_tool, rerank_results

# Load environment variables
load_dotenv()

# Using Google Gemini 3.8 Flash as the RAGAS judge
evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Using Google's embeddings for Answer Relevancy
evaluator_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Define test dataset based on the MongoDB Q4 FY2025 earnings report + out-of-domain tests
test_questions = [
    "What was MongoDB's total revenue for the fourth quarter of fiscal 2025?",
    "How much did MongoDB Atlas revenue grow in the fourth quarter?",
    "What is the customer count for MongoDB as of January 31, 2025?",
    # Out of domain / adversarial questions
    "What is the recipe for chocolate chip cookies?",
    "Who won the FIFA World Cup in 2022?",
    "Can you summarize the plot of the movie Inception?"
]

ground_truths = [
    "MongoDB's total revenue for the fourth quarter was $509.3 million.",
    "MongoDB Atlas revenue grew 30% year-over-year in the fourth quarter.",
    "MongoDB had over 54,100 customers as of January 31, 2025.",
    # Out of domain ground truths
    "I cannot answer this question based on the provided context.",
    "I cannot answer this question based on the provided context.",
    "I cannot answer this question based on the provided context."
]

def run_evaluation():
    print("Preparing data for evaluation...")
    data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": []
    }

    session_id = "eval_session_001"

    for i, question in enumerate(test_questions):
        print(f"\nProcessing question {i+1}: {question}")
        
        # Add a sleep to respect the Voyage AI 3 RPM free tier rate limit
        if i > 0:
            print("Waiting 25 seconds for Voyage AI rate limits (3 RPM max)...")
            time.sleep(25)
            
        # 1. Run your agent to get the answer
        answer = generate_response(session_id, question)
        
        # To avoid a second embedding call, we'll wait another 25 seconds
        print("Waiting 25 seconds before fetching contexts for RAGAS...")
        time.sleep(25)
        
        # 2. Get the contexts that were retrieved
        raw_results = hybrid_search_tool(question)

        if not raw_results:
            contexts = []
        else:
            reranked_results = rerank_results(question, raw_results, top_n=5)
            contexts = [res["text"] for res in reranked_results]
        
        data["user_input"].append(question)
        data["response"].append(answer)
        data["retrieved_contexts"].append(contexts)
        data["reference"].append(ground_truths[i])

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(data)

    print("\nStarting RAGAS evaluation (this may take a minute)...")
    
    # Run evaluation with max_workers=1 to prevent Langchain Google GenAI async timeout bugs
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=1, max_retries=3)
    )

    print("\n--- Evaluation Results ---")
    print(result)
    
    # Print detailed dataframe if you want to inspect row by row
    df = result.to_pandas()
    print("\nDetailed breakdown:")
    print(df[["user_input", "faithfulness", "answer_relevancy", "context_precision"]])

if __name__ == "__main__":
    # Check for keys before running
    if not os.getenv("COHERE_API_KEY") or os.getenv("COHERE_API_KEY") == "YOUR_COHERE_API_KEY_HERE":
        print("ERROR: Please set your COHERE_API_KEY in the .env file before evaluating.")
        exit(1)
        
    run_evaluation()
