# app.py
from flask import Flask, render_template, request, jsonify
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import os

app = Flask(__name__)

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
    if os.path.exists("./storage"):
        print("Loading existing index...")
        # Load the existing index
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = VectorStoreIndex.load_from_storage(storage_context)
    else:
        print("Creating new index...")
        # Load documents and create index
        documents = SimpleDirectoryReader("./data").load_data()
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("cancerdb")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        
        # Save the index
        index.storage_context.persist(persist_dir="./storage")

    query_engine = index.as_query_engine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    response = query_engine.query(user_query)
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    initialize_index()
    app.run(debug=True)