from pymongo import MongoClient
from openai import OpenAI
import voyageai
import cohere
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Environment variables (private)
MONGODB_URI = os.getenv("MONGODB_URI")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# MongoDB cluster configuration
mongo_client = MongoClient(MONGODB_URI)
agent_db = mongo_client["ai_agent_db"]
vector_collection = agent_db["embeddings"]
memory_collection = agent_db["chat_history"]

# Model configuration
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
openai_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
VOYAGE_MODEL = "voyage-4-large"
OPENAI_MODEL = "llama3.2"
COHERE_RERANK_MODEL = "rerank-v3.5"


