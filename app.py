from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from planning import generate_response_ui
from memory import retrieve_session_history
import uvicorn

app = FastAPI(title="Argus Agent API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Process a user message and return the LLM answer along with
    the retrieved sources, rerank scores, and groundedness gate status.
    """
    response_data = generate_response_ui(request.session_id, request.message)
    return response_data

@app.get("/api/history/{session_id}")
def history(session_id: str):
    """
    Retrieve the entire chat history for a session.
    """
    history_data = retrieve_session_history(session_id)
    return {"history": history_data}

if __name__ == "__main__":
    print("Starting Argus API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
