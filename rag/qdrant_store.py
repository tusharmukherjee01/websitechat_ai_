import os
from pathlib import Path

from dotenv import load_dotenv

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    Distance,
    VectorParams
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME"
)


client = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(
        os.getenv("QDRANT_PORT")
    )
)


def create_collection():

    collections = client.get_collections()

    existing = [
        c.name
        for c in collections.collections
    ]

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )