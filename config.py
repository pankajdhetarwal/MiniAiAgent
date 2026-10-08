from pymongo import MongoClient
from openai import OpenAI
import cohere
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Environment variables (private)
MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# MongoDB cluster configuration
mongo_client = MongoClient(MONGODB_URI)
agent_db = mongo_client["ai_agent_db"]
vector_collection = agent_db["embeddings"]
memory_collection = agent_db["chat_history"]

# Model configuration
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
openai_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
OPENAI_MODEL = "llama3.2"
COHERE_RERANK_MODEL = "rerank-v3.5"
