# chat.py - Grounded Conversational Synthesis with History Memory
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
try:
    from .retrieval import retrieve_docs, format_docs, get_prompt_template, get_llm
except ImportError:
    from retrieval import retrieve_docs, format_docs, get_prompt_template, get_llm


def ask_question(question: str, history: List[Dict[str, str]] | None = None, session_id: str = "global") -> Dict[str, Any]:
    """
    Answers a question grounded in the retrieved documents:
    1. Searches vector DB (Supabase or local Chroma) for matching chunks.
    2. Formats retrieved chunks with document sources.
    3. Bundles context, conversation history, and question into the prompt.
    4. Invokes Gemini to generate an answer.
    5. Returns dict with 'answer' and unique 'sources'.
    """
    active_history = history or []

    # 1. Retrieve relevant chunks
    matched_docs = retrieve_docs(question, session_id=session_id)

    # 2. Extract unique source names
    sources: List[str] = []
    for doc in matched_docs:
        src = str(doc.metadata.get("source", "Unknown"))
        if src not in sources:
            sources.append(src)

    # 3. Format chunks into context string
    context_text = format_docs(matched_docs)

    # 4. Convert history format for LangChain (tuples of ('human'/'ai', text))
    chat_history_tuples: List[Tuple[str, str]] = []
    for msg in active_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "human"]:
            chat_history_tuples.append(("human", content))
        elif role in ["assistant", "ai"]:
            chat_history_tuples.append(("ai", content))

    # 5. Build prompt
    prompt_template = get_prompt_template()
    prompt = prompt_template.invoke({
        "chat_history": chat_history_tuples,
        "context": context_text,
        "question": question
    })

    # 6. Generate answer from Gemini LLM
    llm = get_llm()
    response = llm.invoke(prompt)

    # Extract clean text from LLM response (handles str, list of blocks, or AIMessage)
    if isinstance(response.content, str):
        answer_text = response.content
    elif isinstance(response.content, list):
        text_parts: List[str] = []
        for part in response.content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]))
            elif isinstance(part, str):
                text_parts.append(part)
            else:
                text_parts.append(str(part))
        answer_text = "".join(text_parts)
    else:
        answer_text = str(response.content)

    return {
        "answer": answer_text,
        "sources": sources
    }


def start_chat():
    """Terminal CLI interactive chat session."""
    print("\n" + "=" * 55)
    print("  RAG Assistant Started! (Type 'exit' or 'quit' to end)")
    print(f"  Mode: {'Supabase Cloud' if config.USE_SUPABASE else 'Local ChromaDB'}")
    print("=" * 55 + "\n")

    history: List[Dict[str, str]] = []

    while True:
        try:
            user_query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if user_query.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break

        if not user_query:
            continue

        # Get answer using the standardized helper
        res = ask_question(user_query, history=history, session_id="global")

        print(f"\nAssistant:\n{res['answer']}")
        if res["sources"]:
            print(f"\n[Sources: {', '.join(res['sources'])}]\n")
        print("-" * 55)

        # Update history
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": res["answer"]})


if __name__ == "__main__":
    start_chat()
