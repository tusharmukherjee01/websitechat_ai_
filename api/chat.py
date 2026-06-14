from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from rag.embeddings import get_embeddings

from rag.retriever import search_similar

from rag.chain import generate_answer


router = APIRouter()

embeddings = get_embeddings()


class ChatRequest(BaseModel):
    user_id: str
    url: str
    question: str


@router.post("/chat")
def chat(body: ChatRequest):
    try:
        if not body.question or len(body.question) < 3:
            raise HTTPException(400, "Question too short")

        if len(body.question) > 500:
            raise HTTPException(400, "Question too long (max 500 chars)")

        # Embed query
        query_vector = embeddings.embed_query(body.question)

        # Search with user_id filter
        results = search_similar(query_vector, body.user_id, body.url, limit=5)

        if not results:
            return {
                "answer": "I don't have any information about this page yet. Try refreshing or re-indexing."
            }

        # Generate answer
        context = "\n\n".join(r["content"] for r in results)
        answer = generate_answer(body.question, context)

        return {"answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {str(e)}")
