from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ingest import router as ingest_router
from api.chat import router as chat_router

from rag.qdrant_store import create_collection


app = FastAPI(
    title="Website Chat Assistant API"
)


# Allow Chrome Extension requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later replace with your extension origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_collection()


@app.get("/")
def root():
    return {
        "message": "Website Chat Assistant API Running"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

app.include_router(ingest_router)
app.include_router(chat_router)