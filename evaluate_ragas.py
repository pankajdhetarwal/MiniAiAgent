import sys
import types
import os
import uuid

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
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Import the UI version of generate_response — it returns structured data
# including the actual contexts used by the agent (no double-fetch needed)
from planning import generate_response_ui

# Load environment variables
load_dotenv()

# Using Groq's GPT-OSS 120B as the RAGAS judge
# Free tier: 14,400 requests/day, 30 RPM — no more rate limit issues
evaluator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

from langchain_community.embeddings import HuggingFaceEmbeddings

# Using local BGE embeddings for Answer Relevancy (free, offline, no rate limits)
evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
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
    "MongoDB's total revenue for the fourth quarter was $548.4 million.",
    "MongoDB Atlas revenue grew 20% year-over-year in the fourth quarter.",
    "MongoDB had over 54,500 customers as of January 31, 2025.",
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

    # Use a unique session ID per run to prevent session contamination
    # (avoids "you already asked this" responses from the agent's memory)
    session_id = f"eval_{uuid.uuid4().hex[:8]}"
    print(f"Using fresh session: {session_id}")

    for i, question in enumerate(test_questions):
        print(f"\nProcessing question {i+1}/{len(test_questions)}: {question}")
        
        # Use generate_response_ui which returns the answer AND the sources
        # in a single pipeline call — no double-fetching, no extra embedding calls
        result = generate_response_ui(session_id, question)
        
        answer = result["answer"]
        # Extract context texts from the sources the agent actually used
        contexts = [s["text"] for s in result.get("sources", [])]
        
        print(f"  Tool used: {result.get('tool_used', 'N/A')}")
        print(f"  Gate passed: {result.get('gate_passed', 'N/A')}")
        print(f"  Contexts retrieved: {len(contexts)}")
        print(f"  Answer: {answer[:100]}...")
        
        data["user_input"].append(question)
        data["response"].append(answer)
        data["retrieved_contexts"].append(contexts)
        data["reference"].append(ground_truths[i])

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(data)

    print("\n" + "="*60)
    print("Starting RAGAS evaluation with Groq (Llama 3.3 70B)...")
    print("="*60)
    
    # Run evaluation — Groq has 30 RPM so max_workers=1 is safe and avoids async issues
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
    print(df[["user_input", "faithfulness", "answer_relevancy", "context_precision"]].to_string())

if __name__ == "__main__":
    # Check for keys before running
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "YOUR_GROQ_API_KEY_HERE":
        print("ERROR: Please set your GROQ_API_KEY in the .env file before evaluating.")
        print("Get a free key at: https://console.groq.com")
        exit(1)
    if not os.getenv("COHERE_API_KEY") or os.getenv("COHERE_API_KEY") == "YOUR_COHERE_API_KEY_HERE":
        print("ERROR: Please set your COHERE_API_KEY in the .env file before evaluating.")
        exit(1)
        
    run_evaluation()
