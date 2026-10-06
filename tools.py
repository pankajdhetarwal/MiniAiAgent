import ast
import operator as op
from config import vector_collection, cohere_client, COHERE_RERANK_MODEL
from ingest_data import get_embedding


# Define a vector search tool — retrieves top 20 candidates for reranking
def vector_search_tool(user_input: str) -> list:
    query_embedding = get_embedding(user_input, input_type="query")
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": 20,  # fetch more candidates for reranking
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    results = vector_collection.aggregate(pipeline)

    array_of_results = []
    for doc in results:
        array_of_results.append(doc)
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