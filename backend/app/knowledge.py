import os
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ============================================================
# 2. SETTINGS
# ============================================================

COLLECTION_NAME = "solar_pakistan_knowledge"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

FALLBACK = (
    "Ask me about solar panels, inverters, "
    "batteries, sizing, pricing, installation, "
    "or solar energy in Pakistan."
)


# ============================================================
# 3. LOAD CHROMADB
# ============================================================

print("Loading Solar Pakistan knowledge base...")

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)
# Compatibility list for the existing /knowledge endpoint
stored_data = collection.get(
    include=["metadatas"]
)

KNOWLEDGE = []

for metadata in stored_data["metadatas"]:
    KNOWLEDGE.append({
        "topic": "solar",
        "source": metadata.get(
            "source",
            "unknown"
        )
    })

print(
    "ChromaDB loaded successfully!"
)

print(
    "Knowledge chunks:",
    collection.count()
)


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model loaded successfully!"
)


# ============================================================
# 5. FIND ANSWER
# ============================================================

def find_answer(message: str):

    # Create embedding for user's question
    query_embedding = model.encode(
        [message],
        convert_to_numpy=True
    )[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=3
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    # No results
    if not documents:
        return {
            "answer": FALLBACK,
            "topic": "general",
            "sources": []
        }

    # Use the best matching result
    best_document = documents[0]

    best_metadata = (
        metadatas[0]
        if metadatas
        else {}
    )

    best_distance = (
        distances[0]
        if distances
        else None
    )

    # --------------------------------------------------------
    # Basic relevance check
    # --------------------------------------------------------

    # ChromaDB distance is smaller when the result
    # is more similar to the question.
    #
    # If the result is too far away, use fallback.

    if (
        best_distance is not None
        and best_distance > 2.0    ):
        return {
            "answer": FALLBACK,
            "topic": "general",
            "sources": []
        }

    # --------------------------------------------------------
    # Get source
    # --------------------------------------------------------

    source = best_metadata.get(
        "source",
        "unknown"
    )

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {
        "answer": best_document,
        "topic": "solar",
        "sources": [source]
    }