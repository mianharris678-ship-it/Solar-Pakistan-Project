import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


# ============================================================
# 2. SETTINGS
# ============================================================

COLLECTION_NAME = "solar_pakistan_knowledge"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# 3. LOAD MARKDOWN FILES
# ============================================================

def load_markdown_files():

    documents = []

    md_files = glob.glob(
        os.path.join(DATA_DIR, "*.md")
    )

    print(f"\nFound {len(md_files)} Markdown files.")

    for file_path in md_files:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        documents.append({
            "text": text,
            "source": os.path.basename(file_path)
        })

        print("Loaded:", os.path.basename(file_path))

    return documents


# ============================================================
# 4. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for document in documents:

        split_texts = splitter.split_text(
            document["text"]
        )

        for index, text in enumerate(split_texts):

            chunks.append({
                "text": text,
                "source": document["source"],
                "chunk_id": index
            })

    print(f"\nTotal chunks created: {len(chunks)}")

    return chunks


# ============================================================
# 5. GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(chunks):

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded successfully!")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(
        f"\nEmbeddings generated: {len(embeddings)}"
    )

    print(
        f"Embedding dimension: {embeddings.shape[1]}"
    )

    return embeddings


# ============================================================
# 6. STORE IN CHROMADB
# ============================================================

def store_in_chromadb(chunks, embeddings):

    print("\nCreating ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # Delete old collection if it already exists.
    # This prevents duplicate chunks when ingest.py
    # is executed multiple times.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("Old collection removed.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Solar Pakistan knowledge base"
        }
    )

    ids = []
    documents = []
    metadatas = []
    vectors = []

    for index, chunk in enumerate(chunks):

        ids.append(
            f"{chunk['source']}_{chunk['chunk_id']}_{index}"
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        })

        vectors.append(
            embeddings[index].tolist()
        )

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors
    )

    print("\nChromaDB storage completed!")

    print(
        "Total stored chunks:",
        collection.count()
    )

    print(
        "ChromaDB location:",
        CHROMA_DIR
    )


# ============================================================
# 7. TEST RETRIEVAL
# ============================================================

def test_retrieval():

    print("\nTesting retrieval...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    question = "What is solar energy?"

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=3
    )

    print("\nQuestion:", question)

    for i in range(
        len(results["documents"][0])
    ):

        print("\n-----------------------------")

        print(
            f"Result {i + 1}"
        )

        print(
            "Source:",
            results["metadatas"][0][i]["source"]
        )

        print(
            "Distance:",
            results["distances"][0][i]
        )

        print(
            "Content:",
            results["documents"][0][i][:300]
        )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SOLAR PAKISTAN - CHROMADB INGESTION")
    print("=" * 60)

    documents = load_markdown_files()

    if not documents:
        print(
            "\nERROR: No Markdown files found!"
        )
        print(
            "Check the backend/data folder."
        )
        return

    chunks = create_chunks(
        documents
    )

    embeddings = generate_embeddings(
        chunks
    )

    store_in_chromadb(
        chunks,
        embeddings
    )

    test_retrieval()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()