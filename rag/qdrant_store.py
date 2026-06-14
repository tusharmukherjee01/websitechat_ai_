import os
from pathlib import Path

from dotenv import load_dotenv

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME"
)


client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
    )



def create_collection():
    collections = client.get_collections()
    existing = [c.name for c in collections.collections]

    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists")
    else:
        print(f"Creating collection '{COLLECTION_NAME}'")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{COLLECTION_NAME}' created")

    # Ensure payload indexes exist for filtering
    for field in ["user_id", "url", "page_hash"]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD
        )