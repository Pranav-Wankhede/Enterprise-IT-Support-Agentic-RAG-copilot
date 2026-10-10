import time
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}

def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        return 384  # Default to local model dimension

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "all-minilm" in normalized:
        return 384

    return 384  # Fallback dimension for local models

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        model_name = settings.embedding_model or "all-minilm-l6-v2"
        # Using free local HuggingFace embeddings instead of OpenAI
        _embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings

def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    # Sanitize index name to strictly comply with Pinecone rules (lowercase, hyphens only)
    index_name = settings.pinecone_index_name.strip().lower().replace("_", "-")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    names = [x["name"] for x in pc.list_indexes()]

    if index_name in names:
        index_info = pc.describe_index(index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension is not None and current_dimension != desired_dimension:
            pc.delete_index(name=index_name)
            while index_name in [x["name"] for x in pc.list_indexes()]:
                time.sleep(1)

    if index_name not in [x["name"] for x in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(index_name)

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore

def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})

def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)