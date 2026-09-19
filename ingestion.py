import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, DirectoryLoader
# from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
 # all imports are set, hence we now start programming 

def load_docs(docs_path):
    print(f"loading docs nown from {docs_path}...")

    loader = DirectoryLoader( #creating instance of DirectoryLoader in loader object
        path=docs_path,
        glob='*.txt', # used for finding files with the .txt extension
        loader_cls=TextLoader # means we loading text files from the directory
    )

    documents = loader.load() # documents are loaded from the directory

    if(len(documents)==0):
        print("no .txt file in docs")
        raise FileNotFoundError("no .txt file")

    for i, doc in enumerate(documents[:3]):
        print(f"document number {i+1}:")
        print(f"SOURCE: {doc.metadata['source']}")
        print(f"content length : {len(doc.page_content)}")
        print(f" preview : {doc.page_content[:100]}...")
        print("END OF DOCUMENT ")
        print('\n')
        print('\n')
        
    return documents
       
    
# loaded and printed successfully --- end of load ----

#####################################################################


def chunk_docs(docs,size,overlap):
    print("chunking docs now....")

    ##using text splitter to split text very basic
    # for production we would useRecursiveCharacterTextSplitter

    text_splitter=RecursiveCharacterTextSplitter( # or recchartextsplitter.from_tiktkone_encoder orrrr jsut use NLTKTextSplitter - NLP model 
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
        # length_function=len,
        # is_separator_regex=False,
    )

    chunks=text_splitter.split_documents(documents=docs)

    if len(chunks)==0:
        raise ValueError("no chunking happened")

    for i,chunk in enumerate(chunks[:3]):
        print(f"chunk : {i+1}-----")
        print(f"content : {chunk.page_content[:50]}")
        print(f"length of chunk: {len(chunk.page_content)}")
        print(f"metadata : {chunk.metadata}")
        print('\n')
    
    if len(chunks) > 5:
        print(f"and {len(chunks)-i} more chunks created...")
        
    
    return chunks    

###### chunking done using text_splitter


########################################

# now we do embedding n storing the in vector db
def create_vector_db(chunks,storage_path):

    print("starting embeddings and storing in chroma db")

    # 1. Gemini Embedding Model (Commented out):
    # embedding_model=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

    # 2. Local BGE-Small Embedding Model (Active - Zero Rate Limits):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # creating chromadb vector store
    vectordb=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=storage_path,
        collection_metadata={"hnsw:space":"cosine"}
    )
    
    print(f"embedding & storing done, saved to {storage_path}\n")
    
    return vectordb
    
    
def main():
    ###### main func for pipeline

    print("starting pipeline")
    print("1. loading docs...")
    docs_path='docs'
    docs = load_docs(docs_path)
    print("loading done\n")

    ###### main func call for chunking #######

    print("2. chunking docs...")
    size=1500
    overlap=150
    chunks = chunk_docs(docs,size,overlap)
    print("chunking done\n")

    ###### main func call for embedding ##########

    print("3. creating vector db...")
    storage_path="./db"
    vectordb=create_vector_db(chunks,storage_path)
    print("vector db created successfully\n")

    print("INGESTION SUCCESSFULLY DONE \n")

    return vectordb    



if __name__ == "__main__":
    main()    
    









##### to tokenise

# import tiktoken

# tokenizer = tiktoken.get_encoding("cl100k_base")

# def count_tokens(text: str) -> int:
#     return len(tokenizer.encode(text))

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     length_function=count_tokens  # 👈 Now chunk_size=500 measures TOKENS!
# )












