import ast
import operator as op
from config import vector_collection, cohere_client, COHERE_RERANK_MODEL
from ingest_data import get_embedding


# Define a hybrid search tool — retrieves candidates from both vector and text search, merges via RRF
def hybrid_search_tool(user_input: str) -> list:
    # 1. Vector Search
    try:
        query_embedding = get_embedding(user_input, input_type="query")
        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "exact": True,
                    "limit": 20,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        vector_results = list(vector_collection.aggregate(vector_pipeline))
    except Exception as e:
        print(f"Vector search failed: {e}")
        vector_results = []

    # 2. Text Search
    try:
        text_pipeline = [
            {
                "$search": {
                    "index": "text_index",
                    "text": {
                        "query": user_input,
                        "path": "text"
                    }
                }
            },
            {
                "$limit": 20
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "score": {"$meta": "searchScore"},
                }
            }
        ]
        text_results = list(vector_collection.aggregate(text_pipeline))
    except Exception as e:
        print(f"Text search failed: {e}")
        text_results = []

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF score = sum(1 / (60 + rank))
    rrf_scores = {}
    docs_by_id = {}
    
    # Process vector results
    for rank, doc in enumerate(vector_results):
        doc_id = str(doc["_id"])
        docs_by_id[doc_id] = doc["text"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (60 + rank))
        
    # Process text results
    for rank, doc in enumerate(text_results):
        doc_id = str(doc["_id"])
        docs_by_id[doc_id] = doc["text"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (60 + rank))
        
    # Sort by RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 20 combined results
    array_of_results = []
    for doc_id, score in sorted_docs[:20]:
        array_of_results.append({
            "text": docs_by_id[doc_id],
            "rrf_score": score
        })
        
    return array_of_results


# Rerank results using Cohere's cross-encoder model
def rerank_results(query: str, documents: list, top_n: int = 5) -> list:
    """
    Takes the raw vector search results and reranks them using Cohere's
    rerank model. Returns the top_n most relevant documents.
    """
    if not documents:
        return []

    # Extract text from documents for reranking
    doc_texts = [doc["text"] for doc in documents]

    response = cohere_client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=doc_texts,
        top_n=top_n,
    )

    # Build reranked results with Cohere relevance scores
    reranked = []
    for result in response.results:
        original_doc = documents[result.index]
        reranked.append({
            "text": original_doc["text"],
            "vector_score": original_doc.get("score", 0),
            "rerank_score": result.relevance_score,
        })

    return reranked


# Safe arithmetic evaluator — only numbers and +, -, *, /, ** are allowed.
# No function calls, no attribute access, no arbitrary code execution.
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator_tool(user_input: str) -> str:
    try:
        tree = ast.parse(user_input, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"