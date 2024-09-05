# Import necessary libraries
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# Set up Ollama LLM
llm = Ollama(model="qwen2:0.5b", context_window=1024)  # or any other model you have in Ollama

# Configure LlamaIndex to use Ollama
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Load documents from the './data' directory
documents = SimpleDirectoryReader("./data").load_data()

# Parse nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

# Create a Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("cancerdb")

# Construct the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Query the index
response = query_engine.query("Is cancer curable disease?")
print(response)