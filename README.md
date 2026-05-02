# 🚀 VectorDB-mini-GPT

> **A RAG-Powered LLM Chat Assistant** - Intelligent question answering powered by your own data using Vector Databases and Large Language Models.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)
![HNSW](https://img.shields.io/badge/HNSW-Vector%20Search-purple?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-AI%20Pipeline-red?style=for-the-badge)
---

## ✨ Overview

**VectorDB-mini-GPT** is a cutting-edge Retrieval-Augmented Generation (RAG) system that combines the power of Large Language Models with custom vector databases. Upload your documents, ask intelligent questions, and get precise answers grounded in your data—just like ChatGPT, but trained on *your* knowledge base!

### Key Features
- 🎯 **RAG Architecture** - Combines document retrieval with LLM responses
- 📚 **Vector Database Integration** - Fast semantic search over your documents
- 💬 **ChatGPT-like Interface** - Intuitive and user-friendly chat experience
- 🔐 **Data Privacy** - Process your documents locally
- 📤 **Easy Document Upload** - Support for multiple file formats
- ⚡ **Fast Inference** - Optimized for quick responses
- 🎨 **Modern UI** - Clean, responsive design for seamless interaction

---

## 🎯 How It Works


✅ Semantic Search  
✅ Approximate Nearest Neighbor Search  
✅ Interactive Vector Visualization  
✅ Document Embeddings  
✅ AI Question Answering  
✅ Local LLM Inference  

---
Inspired by modern vector systems like:

- Pinecone
- Weaviate
- Chroma
- Milvus

---
## 🔍 Vector Search Algorithms

Implemented from scratch:

### 1. HNSW
Hierarchical Navigable Small World Graph

- Ultra-fast ANN search
- Logarithmic complexity
- Industry standard

### 2. KD-Tree

- Exact nearest neighbor search
- Efficient for low dimensions

### 3. Brute Force

- Exact baseline search
- Ground truth validation

---

# 📚 RAG Pipeline

User Query  
↓  
Embedding Generation  
↓  
HNSW Retrieval  
↓  
Top-k Context Chunks  
↓  
Local LLM Generation  
↓  
Final Answer  

---

# 🧠 AI Models

Powered by Ollama:

### Embedding Model
`nomic-embed-text`

- 768-dimensional embeddings

### Generation Model
`llama3.2`

- Local answer generation

---

# 🛠 Tech Stack

## Backend

- Python
- FastAPI
- NumPy
- Requests
- Pydantic

## AI

- Ollama
- RAG
- Vector Embeddings

## Frontend

- HTML
- CSS
- JavaScript
- Canvas API

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- ~2GB free disk space

### Installation

 **Clone the Repository**
```bash
git clone https://github.com/omkaryadav18/VectorDB-mini-GPT.git
cd VectorDB-mini-GPT
```

##Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

##Install Dependencies
```bash
pip install -r requirements.txt
```
##Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```
##Run Application
```bash
python main.py
Access
Code
http://http://127.0.0.1:8000
```
---

🏗️ **Project Structure**
```bash
VectorDB-mini-GPT/
├── index.html                # Main application
├── requirements.txt          # Dependencies
├── .env             
│   ├── main.py
│   ├── vector_db.py
│   ├── hnsw.py
│   ├── kd_tree.py
│   ├── brute_force.py
│   ├── document_db.py
│   ├── ollama_client.py
│   ├── chunker.py
│   ├── distances.py
│   ├── models.py
```
---


📜 License
MIT License - See LICENSE for details

👨‍💻 Author
Omkar Yadav - @omkaryadav18
