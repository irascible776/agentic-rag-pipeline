## this has 2 parts, retrieveing data chunks and part 2 - feeding data to llm

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

### 2

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

storage_path = "./db"

# 1. Gemini Embedding Model (Commented out):
# embedding_function = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 2. Local BGE-Small Embedding Model (Active - Zero Rate Limits):
embedding_function = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_db = Chroma( #chromadb db object is created here and is been loaded from the ./db folder
                    persist_directory=storage_path,
                    embedding_function=embedding_function,
                    collection_metadata={"hnsw:space":"cosine"}
                    )



#results = vector_db.similarity_search_with_score(query,k=4) # similarity search k- nearest neigbour÷

# here k is the number of chunks we want to retrieve
# distance is the cosine similarity between the query and the chunk
# content is the chunk content
# metadata is the chunk metadata

# fetch_results=vector_db.as_retriever(search_kwargs={"k":4}) # search_kwargs is used to pass arguments to the search method
fetch_results=vector_db.as_retriever(
    search_type='mmr',
    search_kwargs={"k":3, # number of relevant document to return
    "fetch_k":30, # number of chunks top 10 to look at first
    "lambda_mult":0.65 #1 - pure similartity, 0 - max diversity
    },

    ) # search_kwargs is used to pass arguments to the search method


# print(f"{query}\n\n") 

# for i,result in enumerate(search_results):
#     print(f"result {i+1}:\n")
#     print(f"content: {result.page_content}\n\n")

# 2 
llm = ChatGoogleGenerativeAI(
                    model = "gemini-3.6-flash",
                    verbose=True,
                    temperature=0.1 # for consistency
                    )

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[Document {i+1} - Source: {source}]:\n{doc.page_content}")
    return "\n\n".join(formatted)


prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Answer the user's question using ONLY the provided context below.\n"
     "Rules:\n"
     "1. If the answer cannot be found in the context, truthfully say 'I cannot find the answer in the provided documents.'\n"
     "2. Do not invent facts or use outside knowledge.\n"
     "3. Cite which Document number you got each fact from.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])

query = "How did the Wright brothers succeed where others failed?"

print(f"query: {query}\n")

#retrieving chunks
search_results=fetch_results.invoke(query)

#formatting whole doc in one

context_text = format_docs(search_results)


prompt = prompt_template.invoke({
    "context": context_text,
    "question": query
})

response = llm.invoke(prompt)

print(f"\nResponse:\n{response.text}")
