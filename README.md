# Smart Research & Document Assistant

An AI-powered research assistant that allows users to upload PDF documents, extract and index their content, and interactively ask questions using Retrieval-Augmented Generation (RAG). The system provides relevant answers with sources, making document analysis faster and more efficient.

## Features

### 📂 Document Ingestion
- Upload multiple PDF files.
- Extract and segment content using LangChain's PDF loaders and text splitters.

### 🔍 Vector Indexing & Retrieval
- Convert text chunks into embeddings using a vector database (FAISS).
- Retrieve the most relevant sections for accurate responses.

### 🤖 Interactive Q&A Chatbot
- Ask questions about uploaded documents via a user-friendly chat interface.
- Uses an LLM to generate responses based on retrieved document context.

### 🚀 CI/CD Integration
- CI/CD pipeline using GitHub Actions to automate testing and cloud deployment.

## Tech Stack
- **LLM Framework:** LangChain
- **Vector Database:** FAISS
- **CI/CD:** GitHub Actions

## Installation & Setup

### Prerequisites
- Python 3.12
- API keys for LLM models (OpenAI)


### Install Dependencies
```bash
make install
```

### Run the Application
```bash
./main.py
```

## Contributing
Pull requests are welcome! Feel free to fork the repository and submit improvements or bug fixes.

