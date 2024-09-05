# Import necessary libraries
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# Load documents from the './data' directory
documents = SimpleDirectoryReader("./data").load_data()

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Parse nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

# Build embeddings
embeddings = model.encode([node.get_text() for node in nodes])

# Create a Chroma client
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("cancerdb")

# Construct the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Query the vector store
query_engine = storage_context.query_engine()
response = query_engine.query("What is cancer?")
print(response)