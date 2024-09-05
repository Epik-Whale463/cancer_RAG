from flask import Flask, render_template, request, jsonify
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

app = Flask(__name__)

# Set up Ollama LLM
llm = Ollama(model="mistral", context_window=2048, request_timeout=1000)

# Configure LlamaIndex to use Ollama
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Load documents and create index (this could be moved to a separate initialization function)
documents = SimpleDirectoryReader("./data").load_data()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("cancerdb")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
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
    app.run(debug=True)