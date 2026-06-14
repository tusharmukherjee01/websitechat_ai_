from uuid import uuid4
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client.models import PointStruct

from rag.qdrant_store import client, COLLECTION_NAME

from rag.embeddings import get_embeddings


# Optimized for quota management
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

embeddings = get_embeddings()


def batch_embed(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """Embed in batches to avoid rate limits"""
    all_vectors = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)
    
    return all_vectors


def store_page(user_id: str, url: str, title: str, page_hash: str, content: str):
    chunks = splitter.split_text(content)
    
    # Limit maximum chunks per page (quota protection)
    MAX_CHUNKS = 40
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
    
    # Batch embedding
    vectors = batch_embed(chunks)
    
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "user_id": user_id,
                "url": url,
                "title": title,
                "page_hash": page_hash,
                "content": chunk
            }
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search_similar(query_vector: List[float], user_id: str, url: str, limit: int = 5) -> List[dict]:
    """Helper for retrieval with user_id + URL filter"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        query_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="url", match=MatchValue(value=url))
            ]
        )
    ).points
    
    return [{"content": p.payload["content"], "score": p.score} for p in results]
