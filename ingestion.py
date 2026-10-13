# ingestion.py - Multi-Modal Ingestion, Table Extraction & Hierarchical Chunking
import os
import sys
import io
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
from pydantic import SecretStr

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
import media_store
import vision

import pymupdf
import docx
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embedding_model():
    """Returns Gemini embedding model (runs on Google Cloud, uses 0 MB server RAM)."""
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=SecretStr(config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None,
        output_dimensionality=config.EMBEDDING_DIM
    )


def format_markdown_table(rows: List[List[Any]]) -> str:
    """Converts a 2D matrix of cell values into GitHub-Flavored Markdown table syntax."""
    if not rows or len(rows) < 1:
        return ""
    header = [str(c).replace("\n", " ").strip() for c in rows[0]]
    if not any(header):
        return ""

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in rows[1:]:
        clean_row = [str(c).replace("\n", " ").strip() for c in row]
        # Pad row if shorter than header
        if len(clean_row) < len(header):
            clean_row += [""] * (len(header) - len(clean_row))
        lines.append("| " + " | ".join(clean_row[:len(header)]) + " |")

    return "\n".join(lines)


def extract_document_elements(
    filename: str,
    raw_bytes: bytes
) -> Tuple[List[Document], List[Document], List[Document]]:
    """
    Advanced multi-modal extractor for PDF, DOCX, MD, and TXT files.
    Returns three lists of Document objects:
    1. text_docs: Standard narrative prose/paragraphs.
    2. table_docs: Preserved Markdown tables with structural metadata.
    3. image_docs: Extracted figures/diagrams with Gemini Vision summaries and media links.
    """
    name_lower = filename.lower()
    text_docs: List[Document] = []
    table_docs: List[Document] = []
    image_docs: List[Document] = []

    # --- 1. MULTI-MODAL PDF PROCESSING VIA PYMUPDF ---
    if name_lower.endswith(".pdf"):
        try:
            pdf_doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
            stem = Path(filename).stem

            for page_idx in range(len(pdf_doc)):
                page = pdf_doc[page_idx]
                page_num = page_idx + 1

                # A. Extract structured tables first
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        for tab_idx, tab in enumerate(tabs.tables):
                            extracted_rows = tab.extract()
                            if extracted_rows and len(extracted_rows) > 1:
                                md_table = format_markdown_table(extracted_rows)
                                if md_table.strip():
                                    table_docs.append(Document(
                                        page_content=f"### Table: {filename} (Page {page_num}, Table {tab_idx+1})\n\n{md_table}",
                                        metadata={
                                            "source": filename,
                                            "type": "table",
                                            "page": page_num,
                                            "table_index": tab_idx + 1
                                        }
                                    ))
                except Exception as e:
                    print(f"[Ingestion] Table extraction warning on page {page_num}: {e}")

                # B. Extract embedded raster images and diagrams
                try:
                    image_list = page.get_images(full=True)
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        try:
                            base_img = pdf_doc.extract_image(xref)
                            img_bytes = base_img.get("image", b"")
                            img_ext = base_img.get("ext", "png")
                            width = base_img.get("width", 0)
                            height = base_img.get("height", 0)

                            # Filter out tiny icon decorations, spacer pixels, or bullets (<80x80 or <2.5KB)
                            if len(img_bytes) < 2500 or (width < 80 and height < 80):
                                continue

                            # Save image to media store (local disk + optional Supabase bucket)
                            img_filename = f"{stem}_p{page_num}_img{img_idx+1}.{img_ext}"
                            saved_media = media_store.save_media(
                                img_bytes,
                                original_filename=img_filename,
                                mime_type=f"image/{img_ext}"
                            )

                            # Generate semantic description via Gemini Vision
                            description = vision.describe_image(
                                img_bytes,
                                mime_type=f"image/{img_ext}",
                                source_hint=f"{filename} Page {page_num}"
                            )

                            image_docs.append(Document(
                                page_content=f"[Embedded Figure: {filename} (Page {page_num})]\n{description}",
                                metadata={
                                    "source": filename,
                                    "type": "image",
                                    "page": page_num,
                                    "media_path": saved_media["filename"],
                                    "media_url": saved_media["url"],
                                    "media_caption": description[:180]
                                }
                            ))
                        except Exception as img_err:
                            print(f"[Ingestion] Image extract warning on page {page_num}: {img_err}")
                except Exception as e:
                    print(f"[Ingestion] Image inspection warning on page {page_num}: {e}")

                # C. Extract main textual content
                raw_text = page.get_text()
                page_text = str(raw_text) if raw_text else ""
                if page_text.strip():
                    text_docs.append(Document(
                        page_content=page_text.strip(),
                        metadata={"source": filename, "type": "text", "page": page_num}
                    ))


            pdf_doc.close()
        except Exception as e:
            print(f"[Ingestion] Error during PyMuPDF parsing of {filename}: {e}")

    # --- 2. STRUCTURED DOCX PROCESSING VIA PYTHON-DOCX ---
    elif name_lower.endswith(".docx"):
        try:
            doc = docx.Document(io.BytesIO(raw_bytes))
            para_texts = []
            for p in doc.paragraphs:
                if p.text.strip():
                    para_texts.append(p.text.strip())

            if para_texts:
                text_docs.append(Document(
                    page_content="\n\n".join(para_texts),
                    metadata={"source": filename, "type": "text"}
                ))

            # Extract tables into clean Markdown tables
            for tab_idx, table in enumerate(doc.tables):
                rows_data = []
                for row in table.rows:
                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    if any(cells):
                        rows_data.append(cells)
                if rows_data and len(rows_data) > 1:
                    md_table = format_markdown_table(rows_data)
                    if md_table.strip():
                        table_docs.append(Document(
                            page_content=f"### Table: {filename} (Table {tab_idx+1})\n\n{md_table}",
                            metadata={
                                "source": filename,
                                "type": "table",
                                "table_index": tab_idx + 1
                            }
                        ))

            # Extract embedded images from Word DOCX
            stem = Path(filename).stem
            for rel_idx, rel in enumerate(doc.part.related_parts.values()):
                content_type = getattr(rel, "content_type", "")
                if "image" in content_type:
                    img_bytes = getattr(rel, "blob", b"")
                    if len(img_bytes) < 2500:
                        continue
                    ext = content_type.split("/")[-1].replace("jpeg", "jpg")
                    img_name = f"{stem}_img{rel_idx+1}.{ext}"
                    try:
                        saved = media_store.save_media(img_bytes, original_filename=img_name, mime_type=content_type)
                        desc = vision.describe_image(img_bytes, mime_type=content_type, source_hint=filename)
                        image_docs.append(Document(
                            page_content=f"[Embedded Figure: {filename}]\n{desc}",
                            metadata={
                                "source": filename,
                                "type": "image",
                                "media_path": saved["filename"],
                                "media_url": saved["url"],
                                "media_caption": desc[:180]
                            }
                        ))
                    except Exception as img_err:
                        print(f"[Ingestion] Word image extract warning in {filename}: {img_err}")
        except Exception as e:
            print(f"[Ingestion] Error parsing docx {filename}: {e}")


    # --- 3. PLAIN TEXT / MARKDOWN ---
    else:
        try:
            content = raw_bytes.decode("utf-8", errors="ignore").strip()
            if content:
                text_docs.append(Document(
                    page_content=content,
                    metadata={"source": filename, "type": "text"}
                ))
        except Exception as e:
            print(f"[Ingestion] Error decoding text file {filename}: {e}")

    return text_docs, table_docs, image_docs


def extract_text_from_bytes(filename: str, raw_bytes: bytes) -> str:
    """
    Backwards-compatible helper returning plain text representation of the document.
    """
    text_docs, table_docs, _ = extract_document_elements(filename, raw_bytes)
    combined = []
    for d in text_docs:
        combined.append(d.page_content)
    for t in table_docs:
        combined.append(t.page_content)
    return "\n\n".join(combined).strip()


def build_hierarchical_chunks(
    text_docs: List[Document],
    table_docs: List[Document],
    image_docs: List[Document],
    parent_size: Optional[int] = None,
    parent_overlap: Optional[int] = None,
    child_size: Optional[int] = None,
    child_overlap: Optional[int] = None
) -> List[Document]:
    """
    Hierarchical Parent-Child ('Small-to-Big') Chunking:
    1. Splits narrative text into Parent chunks (~1200 chars).
    2. Splits each Parent chunk into Child chunks (~300 chars) for high-accuracy vector search.
    3. Tags each Child chunk with `parent_id` and `parent_content` in its metadata.
    4. Includes tables and image descriptions with self-referential parent context.
    """
    p_size = parent_size or config.PARENT_CHUNK_SIZE
    p_overlap = parent_overlap or config.PARENT_CHUNK_OVERLAP
    c_size = child_size or config.CHILD_CHUNK_SIZE
    c_overlap = child_overlap or config.CHILD_CHUNK_OVERLAP

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=p_size,
        chunk_overlap=p_overlap,
        separators=config.CHUNK_SEPARATORS
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=config.CHUNK_SEPARATORS
    )

    final_chunks: List[Document] = []

    # 1. Process Text Docs through Hierarchical Splitter
    for doc in text_docs:
        source_name = doc.metadata.get("source", "doc")
        page_num = doc.metadata.get("page")
        parent_chunks = parent_splitter.split_documents([doc])

        for p_idx, parent in enumerate(parent_chunks):
            p_content = parent.page_content.strip()
            if not p_content:
                continue

            content_hash = hashlib.md5(p_content.encode("utf-8")).hexdigest()[:8]
            parent_id = f"{source_name}_p{p_idx+1}_{content_hash}"

            # Sub-split into needle-accurate child chunks
            child_chunks = child_splitter.split_documents([parent])
            for c_idx, child in enumerate(child_chunks):
                child_text = child.page_content.strip()
                if not child_text:
                    continue

                child_meta = dict(child.metadata)
                child_meta.update({
                    "source": source_name,
                    "type": "text",
                    "parent_id": parent_id,
                    "parent_content": p_content,
                    "chunk_level": "child",
                    "child_index": c_idx
                })
                if page_num:
                    child_meta["page"] = page_num

                final_chunks.append(Document(
                    page_content=child_text,
                    metadata=child_meta
                ))

    # 2. Add Tables (tables are already compact, so parent = child)
    for t_idx, table_doc in enumerate(table_docs):
        t_meta = dict(table_doc.metadata)
        src = t_meta.get("source", "doc")
        t_meta["parent_id"] = f"{src}_table_{t_idx+1}"
        t_meta["parent_content"] = table_doc.page_content
        t_meta["chunk_level"] = "table"
        final_chunks.append(Document(
            page_content=table_doc.page_content,
            metadata=t_meta
        ))

    # 3. Add Images (vision summaries, parent = child with media links)
    for img_idx, img_doc in enumerate(image_docs):
        i_meta = dict(img_doc.metadata)
        src = i_meta.get("source", "doc")
        i_meta["parent_id"] = f"{src}_image_{img_idx+1}"
        i_meta["parent_content"] = img_doc.page_content
        i_meta["chunk_level"] = "image"
        final_chunks.append(Document(
            page_content=img_doc.page_content,
            metadata=i_meta
        ))

    return final_chunks


def chunk_docs(
    docs: List[Document],
    size: Optional[int] = None,
    overlap: Optional[int] = None
) -> List[Document]:
    """Standard single-level chunking for backward compatibility or flat mode."""
    chunk_size = size or config.CHUNK_SIZE
    chunk_overlap = overlap or config.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config.CHUNK_SEPARATORS
    )
    chunks = text_splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        if "parent_content" not in chunk.metadata:
            chunk.metadata["parent_content"] = chunk.page_content
            chunk.metadata["parent_id"] = f"chunk_{i}"
    return chunks


def load_docs_from_folder(docs_path: Optional[str] = None) -> List[Document]:
    """Loads documents from a local directory, extracting multi-modal elements."""
    folder = Path(docs_path or config.LOCAL_DOCS_DIR)
    print(f"[Ingestion] Loading documents from folder: {folder}...")

    if not folder.exists():
        print(f"[Ingestion] Directory '{folder}' does not exist.")
        return []

    supported_exts = {".txt", ".md", ".pdf", ".docx"}
    all_chunks: List[Document] = []

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in supported_exts:
            continue
        try:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()

            text_docs, table_docs, image_docs = extract_document_elements(file_path.name, raw_bytes)
            file_chunks = build_hierarchical_chunks(text_docs, table_docs, image_docs)
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"[Ingestion] Error reading {file_path.name}: {e}")

    print(f"[Ingestion] Successfully prepared {len(all_chunks)} chunk(s) from folder.")
    return all_chunks


def store_chunks(
    chunks: List[Document],
    session_id: str = "global",
    is_permanent: bool = False
) -> Dict[str, Any]:
    """
    Stores chunks in the vector database:
    - Primary: Supabase pgvector if config.USE_SUPABASE
    - Fallback: Local ChromaDB
    """
    target_session = "global" if is_permanent else session_id
    embed_model = get_embedding_model()

    # --- MODE 1: CLOUD SUPABASE ---
    if config.USE_SUPABASE:
        try:
            print(f"[Ingestion] Cloud Mode: Indexing {len(chunks)} chunks into Supabase table '{config.SUPABASE_TABLE}'...")
            from supabase import create_client
            supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

            # 1. Compute embeddings for all chunk texts
            texts = [chunk.page_content for chunk in chunks]
            embeddings = embed_model.embed_documents(texts)

            # 2. Build rows with session_id tag
            records = []
            for chunk, vec in zip(chunks, embeddings):
                meta = dict(chunk.metadata)
                meta["session_id"] = target_session
                records.append({
                    "session_id": target_session,
                    "content": chunk.page_content,
                    "metadata": meta,
                    "embedding": vec
                })

            # 3. Insert records in batches of 50
            batch_size = 50
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                supabase.table(config.SUPABASE_TABLE).insert(batch).execute()

            print(f"[Ingestion] Successfully stored {len(chunks)} chunks in Supabase under session '{target_session}'.")

            if config.ENABLE_HYBRID_SEARCH:
                try:
                    from hybrid_search import update_bm25_index
                    update_bm25_index(chunks, session_id=target_session)
                except Exception as e:
                    print(f"[Ingestion] Warning updating BM25 index: {e}")

            return {"mode": "supabase", "chunks_stored": len(chunks), "session_id": target_session}
        except Exception as e:

            print(f"[Ingestion] Supabase cloud indexing failed ({e}). Automatically falling back to local ChromaDB...")

    # --- MODE 2: LOCAL CHROMADB FALLBACK ---
    print(f"[Ingestion] Local Mode: Indexing {len(chunks)} chunks into Chroma at {config.LOCAL_DB_DIR}...")
    from langchain_chroma import Chroma

    # Tag metadata with session_id
    for chunk in chunks:
        chunk.metadata["session_id"] = target_session

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=str(config.LOCAL_DB_DIR),
        collection_name="gemini_rag",
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"[Ingestion] Successfully stored {len(chunks)} chunks in local Chroma under session '{target_session}'.")

    # Synchronize BM25 lexical index with new chunks (Phase D)
    if config.ENABLE_HYBRID_SEARCH:
        try:
            from hybrid_search import update_bm25_index
            update_bm25_index(chunks, session_id=target_session)
        except Exception as e:
            print(f"[Ingestion] Warning updating BM25 index: {e}")

    return {"mode": "local", "chunks_stored": len(chunks), "session_id": target_session}




def ingest_documents(
    files_or_folder: Union[str, List[Tuple[str, Any]], None] = None,
    session_id: str = "global",
    is_permanent: bool = False,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Dict[str, Any]:
    """
    Standard unified ingestion pipeline:
    1. Loads files (from disk or memory payload of (filename, bytes_or_str)).
    2. Performs multi-modal extraction (text, tables, and vision summaries).
    3. Builds hierarchical parent-child chunks (or custom flat chunks if slider adjusted).
    4. Embeds & indexes into Supabase or local Chroma.
    """
    if files_or_folder is None:
        files_or_folder = str(config.LOCAL_DOCS_DIR)

    all_chunks: List[Document] = []
    files_count = 0

    if isinstance(files_or_folder, str):
        # Loading from folder
        folder = Path(files_or_folder)
        if not folder.exists():
            return {"status": "error", "message": f"Folder {folder} does not exist.", "chunks_count": 0, "files_count": 0}

        supported_exts = {".txt", ".md", ".pdf", ".docx"}
        for file_path in sorted(folder.iterdir()):
            if file_path.suffix.lower() not in supported_exts:
                continue
            try:
                with open(file_path, "rb") as f:
                    raw_bytes = f.read()
                text_docs, table_docs, image_docs = extract_document_elements(file_path.name, raw_bytes)
                if config.ENABLE_PARENT_CHILD:
                    p_size = chunk_size or config.PARENT_CHUNK_SIZE
                    p_overlap = chunk_overlap or config.PARENT_CHUNK_OVERLAP
                    chunks = build_hierarchical_chunks(
                        text_docs, table_docs, image_docs,
                        parent_size=p_size,
                        parent_overlap=p_overlap,
                        child_size=config.CHILD_CHUNK_SIZE,
                        child_overlap=config.CHILD_CHUNK_OVERLAP
                    )
                else:
                    combined = text_docs + table_docs + image_docs
                    chunks = chunk_docs(combined, size=chunk_size, overlap=chunk_overlap)
                all_chunks.extend(chunks)
                files_count += 1
            except Exception as e:
                print(f"[Ingestion] Error processing {file_path.name}: {e}")

    elif isinstance(files_or_folder, list):
        # Loading from in-memory tuples [(filename, content)]
        for item in files_or_folder:
            if not item or len(item) < 2:
                continue
            filename, content = item[0], item[1]
            try:
                # Convert string content to bytes if passed as string
                raw_bytes = content.encode("utf-8") if isinstance(content, str) else content

                text_docs, table_docs, image_docs = extract_document_elements(filename, raw_bytes)
                if config.ENABLE_PARENT_CHILD:
                    p_size = chunk_size or config.PARENT_CHUNK_SIZE
                    p_overlap = chunk_overlap or config.PARENT_CHUNK_OVERLAP
                    chunks = build_hierarchical_chunks(
                        text_docs, table_docs, image_docs,
                        parent_size=p_size,
                        parent_overlap=p_overlap,
                        child_size=config.CHILD_CHUNK_SIZE,
                        child_overlap=config.CHILD_CHUNK_OVERLAP
                    )
                else:
                    combined = text_docs + table_docs + image_docs
                    chunks = chunk_docs(combined, size=chunk_size, overlap=chunk_overlap)
                all_chunks.extend(chunks)
                files_count += 1
            except Exception as e:
                print(f"[Ingestion] Error processing memory payload {filename}: {e}")


    if not all_chunks:
        return {"status": "error", "message": "No valid documents found to ingest.", "chunks_count": 0, "files_count": 0}

    result = store_chunks(all_chunks, session_id=session_id, is_permanent=is_permanent)

    return {
        "status": "success",
        "files_count": files_count,
        "chunks_count": len(all_chunks),
        "mode": result["mode"],
        "session_id": result["session_id"]
    }


def main():
    """CLI test runner for ingestion."""
    print("=" * 55)
    print("  Running Advanced Multi-Modal Ingestion Pipeline")
    print("=" * 55)
    result = ingest_documents(str(config.LOCAL_DOCS_DIR), session_id="global", is_permanent=True)
    print("\nIngestion Result:", result)


if __name__ == "__main__":
    main()
