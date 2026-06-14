from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from utils.hashing import generate_page_hash

from rag.retriever import store_page

from rag.qdrant_store import client, COLLECTION_NAME

from qdrant_client.models import Filter, FieldCondition, MatchValue


router = APIRouter()


class IngestRequest(BaseModel):
    user_id: str
    title: str
    url: str
    content: str


@router.post("/ingest")
def ingest_page(body: IngestRequest):
    try:
        # Validate content size
        if len(body.content) > 50000:
            raise HTTPException(400, "Content too large (max 50K chars)")

        if len(body.content) < 100:
            raise HTTPException(400, "Content too short (min 100 chars)")

        page_hash = generate_page_hash(body.url, body.content)

        # Efficient dedup: query by user_id + page_hash
        existing = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=body.user_id)),
                    FieldCondition(key="page_hash", match=MatchValue(value=page_hash))
                ]
            ),
            limit=1,
            with_payload=False
        )[0]

        if existing:
            return {"indexed": False, "message": "Already indexed"}

        # Store page with user_id
        store_page(body.user_id, body.url, body.title, page_hash, body.content)

        return {"indexed": True, "message": "Page indexed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Indexing failed: {str(e)}")
