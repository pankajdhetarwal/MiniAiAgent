from config import vector_collection, voyage_client, VOYAGE_MODEL
from pymongo.operations import SearchIndexModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from voyageai.error import APIConnectionError, RateLimitError
import time


# Define a function to generate embeddings, with retry handling for dropped connections and rate limits
def get_embeddings(data, input_type="document", max_retries=5):
    for attempt in range(max_retries):
        try:
            return voyage_client.embed(
                data, model=VOYAGE_MODEL, input_type=input_type
            ).embeddings
        except (APIConnectionError, RateLimitError) as e:
            if attempt == max_retries - 1:
                print(f"Embedding request failed after {max_retries} attempts: {e}")
                raise
            
            # RateLimitError requires a longer wait (60s resets the 1-minute window)
            wait = 60 if isinstance(e, RateLimitError) else 15 * (attempt + 1)
            print(f"Voyage AI Error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}. Retrying in {wait}s...")
            time.sleep(wait)


def get_embedding(data, input_type="document"):
    return get_embeddings([data], input_type=input_type)[0]


# --- Ingest embeddings into MongoDB ---
def ingest_data():
    # Chunk PDF data
    loader = PyPDFLoader("https://investors.mongodb.com/node/13176/pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    documents = text_splitter.split_documents(data)
    print(f"Successfully split PDF into {len(documents)} chunks.")

    # Ingest chunked documents into collection
    print("Generating embeddings and ingesting documents...")
    docs_to_insert = []
    batch_size = 25
    for start in range(0, len(documents), batch_size):
        document_batch = documents[start : start + batch_size]
        embeddings = get_embeddings([doc.page_content for doc in document_batch])
        for doc, embedding in zip(document_batch, embeddings):
            if embedding:
                docs_to_insert.append({"text": doc.page_content, "embedding": embedding})
        if start + batch_size < len(documents):
            print("Waiting for Voyage AI rate limits...")
            time.sleep(25)  # widened from 21s — real breathing room under the 3 RPM cap

    if docs_to_insert:
        result = vector_collection.insert_many(docs_to_insert)
        print(f"Inserted {len(result.inserted_ids)} documents into the collection.")
    else:
        print("No documents were inserted. Check embedding generation process.")

    # --- Create the vector search index ---
    index_name = "vector_index"

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": 1024,
                    "path": "embedding",
                    "similarity": "cosine",
                }
            ]
        },
        name=index_name,
        type="vectorSearch",
    )
    try:
        vector_collection.create_search_index(model=search_index_model)
        print(f"Search index '{index_name}' creation initiated.")
    except Exception as e:
        print(f"Error creating search index: {e}")
        return

    # Wait for initial sync to complete
    print("Polling to check if the index is ready. This may take up to a minute.")
    predicate = lambda index: index.get("queryable") is True

    max_retries = 24  # ~2 minutes at 5s intervals
    for _ in range(max_retries):
        indices = list(vector_collection.list_search_indexes(index_name))
        if len(indices) and predicate(indices[0]):
            break
        time.sleep(5)
    else:
        raise TimeoutError(f"Index '{index_name}' did not become queryable in time.")
    print(index_name + " is ready for querying.")