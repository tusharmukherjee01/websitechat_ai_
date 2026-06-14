from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


def generate_answer(
    question: str,
    context: str
):
    if not context.strip():
        return "Sorry, I couldn't find relevant information to answer your question."

    prompt = f"""You are a website assistant.
Answer only from the provided context. If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"
