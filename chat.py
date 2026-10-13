# chat.py - Grounded Conversational Synthesis with History Memory & Multimodal Grounding
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, cast

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
import media_store
from retrieval import (
    retrieve_docs,
    retrieve_docs_with_telemetry,
    format_docs,
    get_prompt_template,
    get_llm,
    resolve_hierarchical_context,
    extract_media_citations
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def ask_question(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    session_id: str = "global",
    k: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    enable_hybrid: Optional[bool] = None,
    enable_reranker: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Answers a question grounded in the retrieved documents:
    1. Searches vector DB (Supabase or local Chroma) for matching chunks.
    2. Resolves hierarchical parent-child context (deduplicating and expanding).
    3. Extracts media references (diagrams, tables, images).
    4. If images are retrieved, attaches visual Base64 payloads into Gemini's multimodal prompt.
    5. Returns dict with 'answer', 'sources', 'media', and live 'telemetry' latency.
    """
    active_history = history or []

    # 1. Retrieve relevant chunks & track retrieval latency
    t_start = time.perf_counter()
    matched_docs, ret_telem = retrieve_docs_with_telemetry(
        question,
        session_id=session_id,
        k=k,
        enable_hybrid=enable_hybrid,
        enable_reranker=enable_reranker
    )
    t_retrieved = time.perf_counter()
    retrieval_ms = int((t_retrieved - t_start) * 1000)

    # 2. Resolve hierarchical parent context for rich answers without context fragmentation
    context_docs = resolve_hierarchical_context(matched_docs)
    context_text = format_docs(context_docs)

    # 3. Extract unique source names & media citations
    sources: List[str] = []
    for doc in matched_docs:
        src = str(doc.metadata.get("source", "Unknown"))
        if src not in sources:
            sources.append(src)

    media_citations = extract_media_citations(matched_docs)

    # 4. Check if any retrieved images should be visually fed to Gemini
    image_payloads = []
    for citation in media_citations[:2]:  # Limit to top 2 images for low latency
        media_path = citation.get("media_path")
        if media_path:
            try:
                b64_str, mime = media_store.load_media_base64(media_path)
                image_payloads.append({
                    "type": "image_url",
                    "image_url": f"data:{mime};base64,{b64_str}"
                })
            except Exception as e:
                print(f"[Chat] Could not load image for multimodal prompt: {e}")

    # 5. Build prompt (Text or Multimodal)
    llm = get_llm(model=model, temperature=temperature)
    t_gen_start = time.perf_counter()

    if image_payloads:
        # Multimodal invocation with image data URLs
        system_msg = SystemMessage(
            content="You are a helpful and knowledgeable assistant. Answer the user's question using the conversation history, "
                    "the provided text context, and the embedded figures/images.\n"
                    "Rules:\n"
                    "1. If the answer cannot be found in the context or conversation history, truthfully state: 'I cannot find the answer in the provided documents.'\n"
                    "2. Do not invent facts or use outside knowledge.\n"
                    "3. Cite which Document or Figure you referenced."
        )
        msg_history = []
        for msg in active_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ["user", "human"]:
                msg_history.append(HumanMessage(content=content))
            elif role in ["assistant", "ai"]:
                msg_history.append(AIMessage(content=content))

        human_content = [
            {"type": "text", "text": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ]
        human_content.extend(image_payloads)
        current_msg = HumanMessage(content=cast(Any, human_content))


        full_messages = [system_msg] + msg_history + [current_msg]
        response = llm.invoke(full_messages)
    else:
        # Standard text invocation
        chat_history_tuples: List[Tuple[str, str]] = []
        for msg in active_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ["user", "human"]:
                chat_history_tuples.append(("human", content))
            elif role in ["assistant", "ai"]:
                chat_history_tuples.append(("ai", content))

        prompt_template = get_prompt_template()
        prompt = prompt_template.invoke({
            "chat_history": chat_history_tuples,
            "context": context_text,
            "question": question
        })
        response = llm.invoke(prompt)

    t_gen_end = time.perf_counter()
    generation_ms = int((t_gen_end - t_gen_start) * 1000)
    total_ms = int((t_gen_end - t_start) * 1000)

    # Extract clean text from LLM response
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
        "sources": sources,
        "media": media_citations,
        "telemetry": {
            "retrieval_ms": retrieval_ms,
            "dense_search_ms": ret_telem.get("dense_search_ms", 0),
            "bm25_search_ms": ret_telem.get("bm25_search_ms", 0),
            "rerank_ms": ret_telem.get("rerank_ms", 0),
            "candidates_pooled": ret_telem.get("candidates_pooled", len(matched_docs)),
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "chunks_retrieved": len(matched_docs),
            "model_used": model or config.LLM_MODEL,
            "k_used": k or config.RETRIEVER_K
        }
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

        res = ask_question(user_query, history=history, session_id="global")

        print(f"\nAssistant:\n{res['answer']}")
        if res.get("sources"):
            print(f"\n[Sources: {', '.join(res['sources'])}]")
        if res.get("media"):
            media_list = [f"{m['caption']} ({m['url']})" for m in res["media"]]
            print(f"[Visual Citations: {', '.join(media_list)}]")
        print("-" * 55)

        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": res["answer"]})


if __name__ == "__main__":
    start_chat()
