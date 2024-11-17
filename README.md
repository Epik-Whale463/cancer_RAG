
# Cancer RAG Web Application  

## Overview  

The **Cancer RAG Web Application** is a Flask-based web tool that leverages **Retrieval-Augmented Generation (RAG)** using state-of-the-art LLMs and vector search to help users query and retrieve insights about cancer-related information.  
The system combines modern machine learning techniques with a clean and user-friendly interface for a seamless querying experience.  

---  

## Features  

- **RAG Architecture**: Implements Retrieval-Augmented Generation using ChromaDB and LlamaIndex.  
- **Powered by LLMs**: Uses Ollama's `qwen2:0.5b` model and HuggingFace embeddings.  
- **Persistent Vector Store**: Efficient document retrieval with ChromaDB integration.  
- **Customizable Indexing**: Dynamically load cancer-related data to enhance query capabilities.  
- **Interactive UI**: A clean, responsive front-end for entering queries and viewing results.  

---  

## Installation  

Follow these steps to set up and run the application locally:  

### Prerequisites  

1. **Python 3.9+** installed.  
2. **pip** (Python package manager).  

### Steps  

1. Clone the repository:  

   ```bash  
   git clone https://github.com/your_username/cancer-rag-webapp.git  
   cd cancer-rag-webapp  
   ```  

2. Install dependencies:  

   ```bash  
   pip install -r requirements.txt  
   ```  

3. Set up your data directory:  

   - Place your cancer-related documents in a folder named `data` in the project root directory.  

4. Start the application:  

   ```bash  
   python main1.py  
   ```  

5. Open your browser and navigate to:  

   ```  
   http://127.0.0.1:5000  
   ```  

---  

## How It Works  

1. **Data Ingestion**:  
   - Documents placed in the `data/` directory are indexed into a vector store using HuggingFace embeddings and ChromaDB.  

2. **Query Processing**:  
   - When a user inputs a query, the system retrieves relevant information from the vector store and processes it using the `qwen2:0.5b` LLM.  

3. **Interactive Response**:  
   - The response is displayed on the web interface for the user.  

---  

## Project Structure  

```
cancer-rag-webapp/  
├── data/                   # Directory for documents to be indexed.  
├── templates/              # HTML templates for the Flask UI.  
├── static/                 # CSS/JS assets for the UI.  
├── main1.py                # Entry point for the Flask application.  
├── llm.py                  # Logic for indexing and query handling.  
├── requirements.txt        # List of dependencies.  
├── README.md               # Documentation for the project.  
```  

---  

## Sample Output  

1. Navigate to the application in your browser.  
2. Enter a query like:  
   ```  
   What are the common symptoms of lung cancer?  
   ```  
3. The application will display an AI-generated, factually accurate response based on the indexed data.

   ![image](https://github.com/user-attachments/assets/9ae4deaa-1ada-4082-810e-2ea72a4926a0)


---  

## Contributions  

Contributions are welcome! Feel free to submit issues or pull requests.  

---  

## License  

This project is licensed under the MIT License.  
