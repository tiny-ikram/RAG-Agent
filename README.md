# RAG-Agent

## Overview
RAG-Agent is an AI-powered Retrieval-Augmented Generation (RAG) system designed to provide contextual answers and guidance based on technical documentation and guidelines.  
It retrieves relevant documents, processes them, and generates helpful responses using a Large Language Model (LLM).

This implementation is focused on pump troubleshooting and maintenance, using technical manuals and guidelines as the primary knowledge base.

---

## Features
- **Document Retrieval:** Search and retrieve relevant content from PDF guidelines and manuals.
- **Context-Aware Answers:** Use a Large Language Model to generate responses based on retrieved context.
- **Technical Focus:** Optimized for pump troubleshooting with included technical manuals.
- **Interactive Notebook:** Explore and test the RAG pipeline in `RAG_Agent.ipynb`.

---

## Project Structure
RAG-Agent-main/
├── .gitignore # Git ignore rules
├── RAG_Agent.ipynb # Jupyter notebook for experimenting with the RAG pipeline
├── TroubleshootingPumpGuidelines.pdf # Pump troubleshooting guide
├── pumphandbook.pdf # Pump handbook/manual
├── main.py # Main script to run the RAG agent
├── rag_agent.py # RAG Agent class implementation
├── requirements.txt # Python dependencies

---

## Installation

1. **Clone the repository**
git clone https://github.com/your-username/RAG-Agent.git
cd RAG-Agent

2. **Create and activate a virtual environment**

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows


3. **Install dependencies**

pip install -r requirements.txt

Usage
Run from Python script
python main.py

Run in Jupyter Notebook
jupyter notebook RAG_Agent.ipynb

---


How It Works

Load Documents: PDFs are loaded and chunked into smaller sections.

Embedding & Indexing: Text chunks are embedded using an embedding model and stored in a vector database (FAISS).

Query: User query is embedded and matched against the most relevant chunks.

Generate Response: LLM generates a final answer using the retrieved context.
