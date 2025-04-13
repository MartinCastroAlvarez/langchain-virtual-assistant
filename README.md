# LangChain Virtual Assistant

LangChain Virtual Assistant

## Setup

1. Clone the repository:

```bash
git clone https://github.com/MartinCastroAlvarez/langchain-virtual-assistant.git
cd langchain-virtual-assistant
```

2. Install dependencies using Poetry:

```bash
poetry install
```
3. Set the OpenAI API Key.

```bash
export OPENAI_API_KEY="lorem-ipsum"
```

## Usage

1. Place your PDF files in the [./pdfs](./pdfs) directory. Alternatively, you can generate test PDFs using the following command:

```bash
poetry run python3 generator.py --number 10
```

2. You can then generate a vector store in [vectorstore.json](vectorstore.json) . using the following command:

```bash
poetry run python3 indexer.py
```

3. Finally, run the smart agent and start asking questions:

```bash
poetry run python3 agent.py
```
