# gui/server.py - Lightweight Universal Web Server & Pipeline Bridge
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add root directory to sys.path so config and pipeline can be imported directly
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

# import clean, standardized functions from the active pipeline
try:
    from ingestion import ingest_documents as ingest, extract_text_from_bytes
    from chat import ask_question as ask
    from retrieval import clear_session as clear
except ImportError:
    from agentic_rag_pipeline import ingest, ask, clear, extract_text_from_bytes  # type: ignore

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="RAG Studio Web Assistant")

# --- Dynamic CORS Configuration (Localhost + Vercel / Cloud) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS if config.CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "static"


# --- Request Data Models ---
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    session_id: str = "global"


class ClearRequest(BaseModel):
    session_id: str


# --- API Endpoints ---
@app.get("/api/config")
def get_system_config():
    """Returns active storage mode and status for the UI badge."""
    return {
        "mode": "supabase" if config.USE_SUPABASE else "local",
        "supabase_configured": config.USE_SUPABASE,
        "table": config.SUPABASE_TABLE if config.USE_SUPABASE else "local_chroma",
        "embedding_model": config.EMBEDDING_MODEL
    }


@app.post("/api/upload-chunk")
async def upload_and_chunk(
    files: List[UploadFile] = File(...),
    session_id: str = Form("global"),
    is_permanent: bool = Form(False)
):
    """
    Receives uploaded documents (.pdf, .docx, .txt, .md), extracts text into memory,
    and calls the pipeline's ingest() function to chunk and index.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # allowed file extensions
    allowed_exts = (".pdf", ".docx", ".txt", ".md")
    file_tuples: List[Tuple[str, str]] = []

    for f in files:
        if not f.filename or not f.filename.lower().endswith(allowed_exts):
            continue
        try:
            # read uploaded raw bytes into memory
            content_bytes = await f.read()
            # extract clean text according to file format
            text_content = extract_text_from_bytes(f.filename, content_bytes)
            if text_content.strip():
                file_tuples.append((f.filename, text_content))
        except Exception as e:
            print(f"[Server] Error reading upload '{f.filename}': {e}")

    if not file_tuples:
        raise HTTPException(status_code=400, detail="No valid non-empty documents found (.pdf, .docx, .txt, .md).")

    # call the clean pipeline contract
    result = ingest(file_tuples, session_id=session_id, is_permanent=is_permanent)
    return result


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """
    Conversational chat endpoint:
    Calls ask() on the pipeline to retrieve matching chunks and synthesize answer.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = ask(
            question=req.question.strip(),
            history=req.history,
            session_id=req.session_id
        )
        return response
    except Exception as e:
        print(f"[Server] Chat generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/clear")
def clear_endpoint(req: ClearRequest):
    """
    Cleans up uploaded session files and drops session vectors from database.
    Leaves permanent/global docs intact.
    """
    result = clear(req.session_id)
    return result


# --- Static Frontend Serving ---
@app.get("/")
def serve_home():
    return FileResponse(static_dir / "index.html")


app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(f"  Starting RAG Studio on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"  Storage Engine: {'🟢 Supabase Cloud (pgvector)' if config.USE_SUPABASE else '💻 Local ChromaDB (docs/ fallback)'}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
