import hashlib


def generate_page_hash(
    url: str,
    content: str
) -> str:

    raw = f"{url}:{content}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()