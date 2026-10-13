# Open-Source Local & GPU Embedding Models Guide

> **Target Hardware:** MacBook Pro 13-inch (Early 2015)  
> **CPU:** Intel Core i5-5257U (2 cores, 4 threads, AVX2 support)  
> **RAM:** 8 GB LPDDR3 (Shared with iGPU, ~3.6 GB free headroom)  
> **GPU:** Intel Iris Graphics 6100 (48 Execution Units, OpenCL 2.0 / OpenVINO)

---

## 1. Quick Comparison Matrix

All models below are **100% free, open-source (permissive licenses)**, outperform `bge-small` in token limits or dimensions, and can run on your **Intel Iris 6100 GPU** or **AVX2 CPU**.

| Model | Context Window | Dimensions | Weights / RAM | License | Best For |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Nomic Embed Text v1.5** ⭐ *(Top Pick)* | **8,192 tokens** *(16× BGE-Small)* | **768** *(Matryoshka flexible)* | ~280 MB (INT8) / 550 MB | Apache 2.0 | Multi-page PDFs, long documents, flexible storage |
| **2. BAAI / BGE-M3** *(Powerhouse)* | **8,192 tokens** | **1,024** | ~550 MB (INT8) / 1.1 GB | MIT | Hybrid search: Dense + BM25 Sparse + ColBERT |
| **3. Jina Embeddings v2 Base** | **8,192 tokens** | **768** | ~320 MB | Apache 2.0 | Codebases, Markdown docs, technical manuals |
| **4. MixedBread mxbai-embed-large** | 512 tokens | **1,024** | ~670 MB | Apache 2.0 | State-of-the-art semantic precision on short chunks |
| **5. BAAI / BGE-Base v1.5** | 512 tokens | **768** *(2× BGE-Small)* | ~220 MB | MIT | Direct drop-in upgrade for standard chunking |

---

## 2. Top Recommendations & Why

### 🥇 1. `nomic-ai/nomic-embed-text-v1.5` — The Overall Champion
* **Why it's #1:**
  * **8,192 Token Context:** Solves chunk truncation. You can pass entire 15-page documents or long web articles without cutting sentences in half.
  * **Matryoshka Representation Learning (MRL):** The output dimension can be sliced dynamically to **768, 512, 256, or 128 dimensions**. Slicing to 256 dimensions cuts vector database storage and similarity search time by **66%** with less than 1.5% loss in accuracy.
  * **Lightweight:** Only 137M parameters (~280 MB quantized). Loads instantly on the Iris 6100 GPU.
  * **Benchmark:** Beats OpenAI's `text-embedding-ada-002` on the official MTEB retrieval benchmark.

---

### 🥈 2. `BAAI/bge-m3` — The 3-in-1 Hybrid Retrieval Engine
* **Why it's #2:**
  * **Unified Multi-Functionality:** Most RAG pipelines require running two separate tools (e.g. BM25 keyword search + vector dense search). BGE-M3 outputs **both dense vectors (1024-dim) and BM25 sparse weights simultaneously** from a single forward pass.
  * **Multi-Lingual:** Supports over 100 languages with cross-lingual retrieval.
  * **Trade-off:** Heavier model (~560M parameters, ~1.1 GB in memory). It runs well on this machine, but takes slightly longer to embed large batches than Nomic.

---

### 🥉 3. `jinaai/jina-embeddings-v2-base-en` — The Code & Technical Docs Specialist
* **Why it's #3:**
  * **8,192 Token Window for Code:** Specifically trained on GitHub codebases, API documentation, and technical manuals.
  * **Syntax Awareness:** Unlike standard language models that struggle with nested brackets and code syntax, Jina captures functional relationships across classes and functions.
  * **Compact Footprint:** Only ~160M parameters (~320 MB), running easily within system RAM/VRAM constraints.

---

### 4. `mixedbread-ai/mxbai-embed-large` — Pure Short-Chunk Retrieval Accuracy
* **Why choose this:**
  * If your RAG pipeline strictly uses small chunks (e.g., 200–400 words per chunk), this model yields the **highest MTEB accuracy score** in its weight class, rivaling 7-billion parameter models.
  * Native **1,024 dimensions** capture subtle semantic nuances.

---

## 3. How to Run Them on the Intel Iris 6100 GPU

Because the **Intel OpenCL GPU compute runtime** (`intel-opencl-icd`) is active on this system, you can offload inference directly to the Iris 6100 GPU silicon.

### Method A: Intel OpenVINO (Direct Hardware Execution)
Install dependencies:
```bash
pip install optimum[openvino] transformers
```

Run Nomic v1.5 on your Iris 6100 GPU:
```python
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "nomic-ai/nomic-embed-text-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Compiles model for Intel Iris 6100 GPU execution units
model = OVModelForFeatureExtraction.from_pretrained(
    model_id, 
    export=True, 
    device="GPU"
)

text = "Your long document or query here..."
inputs = tokenizer(text, return_tensors="pt", max_length=8192, truncation=True)
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0]  # CLS pooled vector (768 dimensions)
```

---

### Method B: FastEmbed (Lightweight ONNX Runtime)
FastEmbed runs pre-quantized ONNX models with zero PyTorch overhead:
```bash
pip install fastembed
```

```python
from fastembed import TextEmbedding

# Automatically downloads and caches optimized ONNX weights
model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")

documents = [
    "Machine learning pipeline documentation...",
    "System hardware tuning details..."
]

embeddings = list(model.embed(documents))  # Output: 768-dimensional vectors
```

---

## 4. Resource & Memory Safety Check

* **RAM Impact:** Running Nomic v1.5 or Jina uses only **~280 MB to 350 MB** of memory.
* **CPU Load:** Offloading embedding generation to the Iris GPU leaves your dual-core CPU completely unburdened for the synthesis LLM and GUI server.
* **Disk Space:** Model weights occupy between **300 MB and 1.1 GB** on your SSD (you have **50 GB+ free disk space** available).
