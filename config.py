# config.py - Centralized Configuration & Environment Auto-Detection
import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directories
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
PIPELINE_ENV_PATH = ROOT_DIR / "agentic-rag-pipeline" / ".env"

# Load environment variables (check root, then pipeline folder)
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
elif PIPELINE_ENV_PATH.exists():
    load_dotenv(dotenv_path=PIPELINE_ENV_PATH)
else:
    load_dotenv()

# --- 1. Cloud & API Credentials ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

# Check if Supabase is configured; if not, fallback to local storage
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)

# --- 2. Server & Environment Auto-Detection ---
# On Render / cloud hosts, RENDER or PORT is automatically provided
IS_CLOUD = bool(os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT"))
SERVER_HOST = os.getenv("HOST", "0.0.0.0" if IS_CLOUD else "127.0.0.1")
SERVER_PORT = int(os.getenv("PORT", 8000))

# CORS origins (allow any by default locally, or set specific domains in ALLOWED_ORIGINS)
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
CORS_ORIGINS = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

# --- 3. Local Fallback Paths (supabaseurl ?? '/docs') ---
LOCAL_DOCS_DIR = ROOT_DIR / "docs" if (ROOT_DIR / "docs").exists() else ROOT_DIR / "agentic-rag-pipeline" / "docs"
LOCAL_DB_DIR = ROOT_DIR / "db" if (ROOT_DIR / "db").exists() else ROOT_DIR / "agentic-rag-pipeline" / "db"
LOCAL_MEDIA_DIR = LOCAL_DB_DIR / "media"
LOCAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# --- 4. LLM, Vision & Embedding Settings ---
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.6-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))

# Gemini embedding model (768 dimensions via output_dimensionality=768, 0 MB server RAM)
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# --- 5. Chunking & Splitting Settings (Level 1 & Level 2 Parent-Child) ---
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ".", " "]

# Level 2 Hierarchical Parent-Child Chunking
ENABLE_PARENT_CHILD = os.getenv("ENABLE_PARENT_CHILD", "true").lower() in ("true", "1", "yes")
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", 1200))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", 150))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", 300))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", 50))

# --- 6. Retrieval & Search Settings ---
RETRIEVER_K = 5
FETCH_K = 30
LAMBDA_MULT = 0.65

# --- 7. Supabase Database & Storage Constants ---
SUPABASE_TABLE = "documents"
SUPABASE_MEDIA_BUCKET = os.getenv("SUPABASE_MEDIA_BUCKET", "rag-media")

# --- 8. Level 3: Hybrid Search & Two-Stage Re-Ranking Settings ---
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() in ("true", "1", "yes")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() in ("true", "1", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")
HYBRID_FETCH_K = int(os.getenv("HYBRID_FETCH_K", 25))
RRF_SMOOTHING = int(os.getenv("RRF_SMOOTHING", 60))
BM25_INDEX_PATH = LOCAL_DB_DIR / "bm25_index.pkl"

