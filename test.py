import time
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Sample test data: 1 query, 1 matching document, 1 irrelevant document
query = "How did the Wright brothers control aircraft roll?"
doc_relevant = "The Wright brothers invented wing-warping to control the roll of their glider."
doc_irrelevant = "A bicycle has two wheels and is powered by pedals."

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_model(model_name, model_kwargs={}):
    print(f"\n{'=' * 60}")
    print(f"Testing Model: {model_name}")
    print(f"{'=' * 60}")

    # 1. Measure load time
    start_load = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu", **model_kwargs},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"Loaded in: {time.time() - start_load:.2f} seconds")

    # 2. Measure embedding time
    start_embed = time.time()
    query_vec = embeddings.embed_query(query)
    doc_match_vec = embeddings.embed_documents([doc_relevant])[0]
    doc_other_vec = embeddings.embed_documents([doc_irrelevant])[0]
    print(f"Embedded 3 texts in: {time.time() - start_embed:.3f} seconds")

    # 3. Check vector dimensions
    print(f"Vector Dimensions: {len(query_vec)}")

    # 4. Check semantic similarity scores
    score_match = cosine_similarity(query_vec, doc_match_vec)
    score_other = cosine_similarity(query_vec, doc_other_vec)

    print(f"Similarity with Relevant Doc:   {score_match:.4f} (Higher = Better)")
    print(f"Similarity with Irrelevant Doc: {score_other:.4f} (Lower = Better)")

# --- RUN TESTS ---

# Test 1: BAAI/bge-small-en-v1.5 (512 token limit, 384 dimensions)
test_model("BAAI/bge-small-en-v1.5")

# Test 2: nomic-ai/nomic-embed-text-v1.5 (8,192 token limit, 768 dimensions)
# Note: nomic requires trust_remote_code=True
test_model("nomic-ai/nomic-embed-text-v1.5", model_kwargs={"trust_remote_code": True})
