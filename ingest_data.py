from config import vector_collection
from pymongo.operations import SearchIndexModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load the local BGE embedding model once at module level
# BAAI/bge-base-en-v1.5: 768-dim, no API key, no rate limits
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def get_embeddings(data, input_type="document"):
    """Generate embeddings for a list of texts using the local BGE model."""
    # BGE recommends prepending "Represent this sentence:" for retrieval queries
    if input_type == "query":
        data = [f"Represent this sentence: {text}" for text in data]
    return embedding_model.encode(data).tolist()


def get_embedding(data, input_type="document"):
    """Generate a single embedding."""
    return get_embeddings([data], input_type=input_type)[0]


# --- Ingest embeddings into MongoDB ---
def ingest_data():
    # Chunk PDF data
    loader = PyPDFLoader("MongoDB, Inc. Announces Fourth Quarter and Full Year Fiscal 2025 Financial Results.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    documents = text_splitter.split_documents(data)
    print(f"Successfully split PDF into {len(documents)} chunks.")

    # Ingest chunked documents into collection
    print("Generating embeddings and ingesting documents...")
    docs_to_insert = []
    batch_size = 50  # Local model — no rate limits, larger batches are fine
    for start in range(0, len(documents), batch_size):
        document_batch = documents[start : start + batch_size]
        embeddings = get_embeddings([doc.page_content for doc in document_batch])
        for doc, embedding in zip(document_batch, embeddings):
            if embedding:
                docs_to_insert.append({"text": doc.page_content, "embedding": embedding})
        print(f"  Embedded batch {start // batch_size + 1}")

    if docs_to_insert:
        result = vector_collection.insert_many(docs_to_insert)
        print(f"Inserted {len(result.inserted_ids)} documents into the collection.")
    else:
        print("No documents were inserted. Check embedding generation process.")

    # --- Create the vector and text search indices ---
    vector_index_name = "vector_index"
    text_index_name = "text_index"

    vector_search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": 768,  # BGE-base-en-v1.5 produces 768-dim vectors
                    "path": "embedding",
                    "similarity": "cosine",
                }
            ]
        },
        name=vector_index_name,
        type="vectorSearch",
    )
    
    text_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True
            }
        },
        name=text_index_name,
        type="search",
    )
    
    try:
        vector_collection.create_search_index(model=vector_search_index_model)
        print(f"Search index '{vector_index_name}' creation initiated.")
        vector_collection.create_search_index(model=text_search_index_model)
        print(f"Search index '{text_index_name}' creation initiated.")
    except Exception as e:
        print(f"Error creating search index: {e}")
        return

    # Wait for initial sync to complete
    import time
    print("Polling to check if the indices are ready. This may take up to a minute.")
    predicate = lambda index: index.get("queryable") is True

    max_retries = 24  # ~2 minutes at 5s intervals
    for _ in range(max_retries):
        v_indices = list(vector_collection.list_search_indexes(vector_index_name))
        t_indices = list(vector_collection.list_search_indexes(text_index_name))
        if len(v_indices) and predicate(v_indices[0]) and len(t_indices) and predicate(t_indices[0]):
            break
        time.sleep(5)
    else:
        raise TimeoutError("Indices did not become queryable in time.")
    print("Vector and text search indices are ready for querying.")