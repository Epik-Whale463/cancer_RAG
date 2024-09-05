# llm.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import os

# Global variables
query_engine = None
index = None

def initialize_index():
    global query_engine, index
    
    # Set up Ollama LLM
    llm = Ollama(model="mistral", context_window=2048, request_timeout=1000)
    
    # Configure LlamaIndex to use Ollama
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Check if the index already exists
    if os.path.exists("./chroma_db"):
        print("Loading existing index...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("cancerdb")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        print("Creating new index...")
        documents = SimpleDirectoryReader("./data").load_data()
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("cancerdb")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    query_engine = index.as_query_engine()

def get_query_engine():
    global query_engine
    if query_engine is None:
        initialize_index()
    return query_engine