# Advanced RAG System Architecture: Level 2 & Level 3 Playbook

> **Target Objective:** Evolve the current naive RAG system into a production-grade, multi-modal, hybrid search and re-ranked retrieval pipeline.  
> **Accuracy Target:** Elevate retrieval precision from ~70% to **95–97%**.  
> **Hardware Footprint:** **0 MB local GPU RAM overhead** (leverages Gemini Vision API and lightweight CPU re-ranking).

---

## 🗺️ System Evolution Map

```
CURRENT STATUS (Level 1: Naive RAG) [~70% Accuracy]
  Text-only • Fixed 1500-char chunks • Flat vector search • Blind to images & tables
                              │
                              ▼
NEXT PHASE (Level 2: Multi-Modal & Hierarchical) [~90–93% Accuracy]
  1. Image & diagram extraction with Gemini Vision
  2. Structured Markdown table preservation
  3. Parent-Child ("Small-to-Big") chunking
  4. Disk / Supabase Storage + on-the-fly Base64 loading
  5. Visual citations rendered in Chat UI
                              │
                              ▼
FINAL PHASE (Level 3: Hybrid Search & Re-Ranking) [~95–97% Accuracy]
  1. Dense Vector search (semantics) + Sparse BM25 (exact keywords)
  2. Reciprocal Rank Fusion (RRF) score merging
  3. Two-Stage Cross-Encoder Re-Ranking (FlashRank / Cohere)
```

---

## 🧠 Part 1: Deep Explanations (Simple vs. Technical)

---

### LEVEL 2: Multi-Modal Ingestion & Hierarchical Architecture

#### Component 2.1: Multi-Modal Image & Diagram Ingestion
* **In Simple Terms:**
  Right now, if you upload a PDF containing a sales chart or an architecture diagram, the system throws the image in the trash and only reads the text around it. It is literally blind to pictures.  
  *The Fix:* When the system spots an image, it saves the picture in a safe storage locker (a folder on disk or a Supabase bucket). Then, it shows the picture to Gemini in the cloud and asks: *"Explain everything in this diagram in plain English."* We store that text explanation in the search database. When a user asks about the diagram, we find the explanation, fetch the original image from the locker, convert it to an image payload in memory, send it to Gemini so it can double-check the visual details with its own eyes, and show the original diagram directly to the user in chat.
* **In Technical Terms:**
  * **Extraction:** Ingestion pipeline uses `pymupdf` (fitz) to extract embedded raster images and bounding coordinates.
  * **Dual-Layer Storage:** The binary image payload is written to `db/media/<hash>.png` (Local) or uploaded to Supabase Storage bucket `rag-media` (Cloud).
  * **Semantic Summarization:** The image is passed to `gemini-1.5-flash` / `gemini-2.0-flash` with a structured system prompt extracting visual entities, flow directions, axis labels, and numerical metrics.
  * **Vector Indexing:** The generated summary is vectorized (`gemini-embedding-001`, 768 dims) and stored in Chroma/Supabase with metadata: `{"type": "image", "media_path": "...", "source": "..."}`.
  * **On-Demand Hydration:** At retrieval time, the matching document's `media_path` is read, converted to Base64 in memory, passed into Gemini's multimodal message array (`image_url`), and emitted to the client UI as a visual citation.

---

#### Component 2.2: Structured Table Preservation
* **In Simple Terms:**
  When a standard text reader looks at a spreadsheet or table in a PDF, it reads it left-to-right like a sentence. A table with 3 columns (`Product`, `Price`, `Stock`) turns into `Product Price Stock Apple $2 50 Orange $3 20`—a garbled mess where the AI loses track of which number belongs to which item.  
  *The Fix:* We extract tables as clean Markdown tables with rows and columns intact (`| Product | Price | Stock |`). We also have Gemini write a short 2-sentence summary of the table trends for fast search.
* **In Technical Terms:**
  * **Layout Detection:** Identifies table cells and coordinates using `pdfplumber` or `python-docx` table objects.
  * **Serialization:** Serializes tabular matrices into GitHub-Flavored Markdown syntax, preserving header boundaries.
  * **Dual-Representation Indexing:** Generates an embedding for the table's natural language summary while retaining the raw Markdown matrix in the chunk payload for exact LLM context synthesis.

---

#### Component 2.3: Hierarchical Parent-Child Chunking ("Small-to-Big")
* **In Simple Terms:**
  *The Dilemma:* If you cut text into tiny snippets (1 sentence), the search engine easily finds the exact match, but the AI writing the answer doesn't have enough surrounding context to understand what happened. If you cut text into massive blocks (3 pages), the AI has plenty of context, but the search engine struggles to find specific details because the vector gets diluted.  
  *The Fix:* **Small for searching, Big for answering.** We chop the document into tiny 150-character "Child" chunks and put them in the search database. When a search hits a child chunk, the system secretly looks up the larger 1,000-character "Parent" section that the child belongs to and feeds that full parent block to the AI.
* **In Technical Terms:**
  * **Child Chunks (100–150 tokens):** High semantic specificity, optimized for maximum vector cosine similarity and needle-in-haystack recall.
  * **Parent Chunks (800–1,200 tokens):** Stored in a key-value Docstore (Chroma docstore or relational table).
  * **Resolution:** Retriever queries the vector collection containing child embeddings; maps child chunk `parent_id` foreign keys to the parent docstore, resolving the context window penalty without sacrificing recall precision.

---

#### Component 2.4: Visual Citations in the Web UI
* **In Simple Terms:**
  Instead of the chatbot just saying: *"The company profits rose 15% in Q3"*, it shows the exact bar chart image directly below the text answer with a small tag: *"Source: Page 4, Financial Report"*. It gives the user instant visual proof.
* **In Technical Terms:**
  * **Payload Augmentation:** The chat API endpoint returns an array of media citations alongside the generated token stream: `{ answer: string, sources: string[], media: [{ url: string, type: "image", caption: string }] }`.
  * **DOM Injection:** Frontend `app.js` dynamically creates visual citation cards with lightbox expansion modals.

---

### LEVEL 3: Hybrid Search & Two-Stage Re-Ranking

#### Component 3.1: Hybrid Search (Dense Vectors + Sparse BM25 + RRF)
* **In Simple Terms:**
  Vector search is smart about meanings (it knows that *"automobile"* and *"car"* are the same thing), but it is terrible at exact codes, acronyms, and part numbers (e.g. searching for error code `0x80040154` or product `RTX-4090Ti` often returns random tech articles).  
  Keyword search (BM25) is the opposite: it doesn't understand meaning, but it never misses an exact keyword match.  
  *The Fix:* We run **both searches simultaneously**. We search by meaning AND by exact words, then merge the scores using an algorithm called **RRF (Reciprocal Rank Fusion)** so the best of both worlds floats to the top.
* **In Technical Terms:**
  * **Dense Retrieval:** Computes cosine distance against `gemini-embedding-001` 768-dimensional vectors.
  * **Sparse Retrieval:** Tokenizes corpus and queries using an inverted index with **BM25Okapi** scoring (evaluating term frequency and inverse document frequency).
  * **Reciprocal Rank Fusion (RRF):** Merges the ranked lists without needing normalized score distributions:
    $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    *(where $r_m(d)$ is the rank of document $d$ in search method $m$, and $k \approx 60$ is a smoothing constant).*

---

#### Component 3.2: Two-Stage Cross-Encoder Re-Ranking
* **In Simple Terms:**
  Think of vector search as a **speed dating round**: it looks through 10,000 documents in 5 milliseconds and picks the top 30 candidates that look roughly okay.  
  Then, the **Re-Ranker** acts like a **strict professor**: it takes only those 30 candidates, reads every single word of the question and the document side-by-side with full attention, and grades them from 0 to 100. It then reorders them so the single absolute best document is guaranteed to be #1.
* **In Technical Terms:**
  * **Bi-Encoder (Stage 1 - Recall):** Compares pre-computed query vectors against document vectors independently. Blazingly fast, but loses fine-grained cross-token interactions.
  * **Cross-Encoder (Stage 2 - Precision):** Passes the Query and Candidate Document simultaneously through all transformer layers:
    $$\text{Score} = \text{Softmax}(\text{Linear}(\text{Transformer}([Query, Document])))$$
    Every token in the query attends to every token in the document.
  * **Engine Choice:** **FlashRank** (Ultra-lightweight, quantized ONNX model, runs in <15ms on CPU with zero GPU dependencies and ~40 MB memory footprint).

---

## 📋 Comprehensive Master Task List

### 📦 PHASE A: Media Storage & Parsing Infrastructure (Level 2)
- [x] **A.1 Media Storage Manager (`media_store.py`)**
  - Implement dual storage: Local folder `db/media/` and Supabase Storage bucket `rag-media`.
  - Add image helper functions: file saving, mime-type verification, and fast in-memory Base64 encoding.
- [x] **A.2 Advanced Document Extractor (`ingestion.py`)**
  - Integrate `pymupdf` for PDF parsing.
  - Extract text blocks, tables, and raw image streams separately with page-number metadata.
  - Implement table extraction into clean GitHub-Flavored Markdown.
- [x] **A.3 Gemini Vision Summarizer (`vision.py`)**
  - Connect Gemini Vision to inspect extracted images and diagrams.
  - Prompt: Extract core message, data points, labels, and diagram flows for search indexing.
  - Save image to media storage; generate searchable text chunk linked to the image file.

---

### 🌲 PHASE B: Hierarchical Parent-Child Chunking (Level 2)
- [x] **B.1 Parent-Child Splitter Implementation**
  - Configure `child_splitter`: 300 characters, 50 overlap.
  - Configure `parent_splitter`: 1,200 characters, 150 overlap.
  - Store child chunks in Vector DB (Chroma/Supabase) with `parent_id` and `parent_content` in metadata.
- [x] **B.2 Hierarchical Retriever Resolver (`retrieval.py`)**
  - Search vector database for child matches.
  - Deduplicate and resolve `parent_id` keys to fetch complete parent blocks for the LLM.

---

### 🎨 PHASE C: Multimodal Chat & UI Citations (Level 2)
- [x] **C.1 Multimodal LLM Synthesis (`chat.py`)**
  - If retrieved context includes an image chunk:
    - Load image from media store into memory as Base64.
    - Append visual image payload into Gemini's multimodal prompt.
  - Return answer along with structured media references (`media_citations`).
- [x] **C.2 Web GUI Visual Citations (`gui/server.py` & `gui/static/app.js` & `gui/static/style.css`)**
  - Add static media endpoint `/media/{filename}` mounted on FastAPI server.
  - Update `app.js` and `style.css` to render image preview cards and expandable modal lightboxes below answers.


---

### 🔍 PHASE D: Hybrid Search (Level 3)
- [x] **D.1 BM25 Lexical Index (`hybrid_search.py`)**
  - Build in-memory or persisted BM25 inverted index using `rank-bm25`.
  - Synchronize index updates on every document ingestion.
- [x] **D.2 Reciprocal Rank Fusion (RRF)**
  - Implement `reciprocal_rank_fusion(dense_results, sparse_results, k=60)`.
  - Combine dense vector results and BM25 results into unified candidate list of top 30 documents.

---

### 🎯 PHASE E: Two-Stage Re-Ranking (Level 3)
- [x] **E.1 Integrate Lightweight Re-Ranker (`reranker.py`)**
  - Add `flashrank` dependency (`pip install flashrank`).
  - Initialize lightweight ONNX cross-encoder model (`ms-marco-TinyBERT-L-2-v2` or `bge-reranker-small`).
- [x] **E.2 Integrate into Retrieval Pipeline (`retrieval.py`)**
  - Top 30 candidates from Hybrid RRF $\rightarrow$ FlashRank $\rightarrow$ Return top 5 highest-confidence chunks to LLM.
- [x] **E.3 Live Telemetry & Benchmarking**
  - Track and display search latency breakdown in UI: Vector Search ms, BM25 ms, Re-ranking ms, and LLM Synthesis ms.

---

## 📖 Glossary: Plain English to Technical Terms Dictionary

| Plain English Concept | Technical Industry Term | What It Actually Means |
| :--- | :--- | :--- |
| **Search by meaning** | **Dense Vector Retrieval** | Turning sentences into numbers (embeddings) and measuring the angle/distance between them. |
| **Search by exact word** | **Sparse Lexical Retrieval (BM25)** | Counting exact keyword matches with statistical adjustments for common words. |
| **Merging two searches** | **Reciprocal Rank Fusion (RRF)** | A math formula that combines two different leaderboard rankings into a single fair score. |
| **The strict grading judge** | **Cross-Encoder Re-Ranker** | A deep learning model that reads the question and document simultaneously to grade relevance. |
| **Search small, feed big** | **Hierarchical Parent-Child Chunking** | Indexing tiny chunks for search accuracy, but feeding large parent paragraphs to the LLM. |
| **Extracting picture meaning** | **Multi-Modal Vision Indexing** | Passing images to an AI vision model to create text summaries that can be searched. |
| **Image as text string** | **Base64 Encoding** | Converting a binary `.png` or `.jpg` file into ASCII text so it can travel over APIs and HTML. |
| **Visual evidence** | **Visual Grounding / Visual Citation** | Showing the user the real diagram or chart cited by the AI as proof of the answer. |
| **Two-step filter** | **Two-Stage Retrieve-and-Rerank** | Pulling a broad set of candidates quickly (recall), then scoring them with a heavy model (precision). |
