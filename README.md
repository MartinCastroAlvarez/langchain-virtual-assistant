# LangChain Virtual Assistant

A virtual assistant built using LangChain that can process PDFs, perform web searches, and answer questions using RAG (Retrieval-Augmented Generation).

## Features

- PDF processing and text extraction
- Web search capabilities
- RAG-based question answering
- Conversation memory
- Vector embeddings for semantic search
- PDF caching for faster processing

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MartinCastroAlvarez/langchain-virtual-assistant.git
cd langchain-virtual-assistant
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Copy the environment variables template and fill in your API keys:

```bash
cp .env.example .env
```

## Usage

1. Place your PDF files in the [./pdfs](./pdfs) directory
2. Run the assistant:
```bash
poetry run python src/langchain_agent_project/agent.py
```

3. Interact with the assistant in the terminal. Type 'salir', 'exit', or 'quit' to end the session.

## Project Structure

```
langchain_agent_project/
├── .env
├── pyproject.toml
├── README.md
├── src/
│   └── langchain_agent_project/
│       ├── __init__.py
│       └── agent.py
└── pdfs/
```

## Dependencies

- Python 3.9+
- LangChain
- OpenAI API
- FAISS
- PyPDF
- BeautifulSoup4
- Requests
- Transformers (optional, for local models)

## License

MIT


