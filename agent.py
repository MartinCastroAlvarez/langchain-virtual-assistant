import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.tools import tool
from langchain.prompts import PromptTemplate

import tempfile
import pickle
import glob
from typing import List
from bs4 import BeautifulSoup
import requests

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

PDF_DIR = "pdfs"
CACHE_DIR = "/tmp/pdf_cache"
CONV_PATH = "/tmp/convo.pkl"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, memory_key="chat_history")

# Cache
def get_embedding_cache_path(pdf_name):
    return os.path.join(CACHE_DIR, f"{pdf_name}.pkl")

# Load PDFs and create vectorstore
def load_pdf_embeddings() -> FAISS:
    docs = []
    for pdf_path in glob.glob(f"{PDF_DIR}/*.pdf"):
        pdf_name = os.path.basename(pdf_path)
        cache_file = get_embedding_cache_path(pdf_name)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                partial_index = pickle.load(f)
        else:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            partial_index = FAISS.from_documents(pages, OpenAIEmbeddings())
            with open(cache_file, "wb") as f:
                pickle.dump(partial_index, f)
        docs.extend(partial_index.docstore._dict.values())
    return FAISS.from_documents(docs, OpenAIEmbeddings())

vectorstore = load_pdf_embeddings()

# Tools
@tool
def search_web(query: str) -> str:
    """Hace búsquedas simples en internet y retorna el contenido como texto."""
    url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    return soup.get_text()[:2000]  # limitar la salida

@tool
def list_pdfs(_: str) -> str:
    """Lista los nombres de los PDFs disponibles en el directorio."""
    return "\n".join(os.listdir(PDF_DIR))

@tool
def convert_pdf_to_text(pdf_name: str) -> str:
    """Convierte un PDF específico a texto."""
    path = os.path.join(PDF_DIR, pdf_name)
    loader = PyPDFLoader(path)
    return "\n".join([p.page_content for p in loader.load()])

@tool
def embed_question(text: str) -> List[float]:
    """Genera el embedding de una pregunta."""
    return OpenAIEmbeddings().embed_query(text)

@tool
def similarity_search(query: str) -> str:
    """Hace una búsqueda por similitud en los documentos PDF embebidos."""
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

# RAG Tool
@tool
def answer_with_rag(query: str) -> str:
    """Hace una pregunta basada en los PDFs embebidos (RAG)."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa_chain.run(query)

# Agente
tools = [
    search_web,
    list_pdfs,
    convert_pdf_to_text,
    embed_question,
    similarity_search,
    answer_with_rag,
]

# Restaurar memoria
if os.path.exists(CONV_PATH):
    with open(CONV_PATH, "rb") as f:
        memory.chat_memory = pickle.load(f)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory,
)

# Ejecución
def run_agent():
    print("🧠 Asistente iniciado. Escribe 'salir' para terminar.")
    while True:
        query = input("Tú: ")
        if query.lower() in ["salir", "exit", "quit"]:
            break
        response = agent.run(query)
        print("Asistente:", response)

    # Guardar conversación
    with open(CONV_PATH, "wb") as f:
        pickle.dump(memory.chat_memory, f)

if __name__ == "__main__":
    run_agent() 