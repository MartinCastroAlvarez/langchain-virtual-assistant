from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import requests
from bs4 import BeautifulSoup
from colorama import Fore
from colorama import Style
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LC_Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in environment variables."

PDF_DIR: str = "./pdfs"
VECTORSTORE_FILE: str = "./vectorstore.json"
CONVERSATION_CACHE: str = "/tmp/conversation_cache.pkl"
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
CACHE_DIR: str = "/tmp/sentence_transformers"


class Out:
    @staticmethod
    def green(message: str) -> None:
        print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")

    @staticmethod
    def red(message: str) -> None:
        print(f"{Fore.RED}{message}{Style.RESET_ALL}")

    @staticmethod
    def cyan(message: str) -> None:
        print(f"{Fore.CYAN}{message}{Style.RESET_ALL}")

    @staticmethod
    def yellow(message: str) -> None:
        print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")

    @staticmethod
    def blue(message: str) -> None:
        print(f"{Fore.BLUE}{message}{Style.RESET_ALL}")

    @staticmethod
    def white(message: str) -> None:
        print(f"{Fore.WHITE}{message}{Style.RESET_ALL}")

    @staticmethod
    def input(prompt: str = "") -> str:
        return input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}")


class Vector:
    model: SentenceTransformer | None = None

    @classmethod
    def load(cls) -> None:
        cls.model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=CACHE_DIR)
        Out.green(f"Vector model loaded with {len(cls.model.encode('test'))} dimensions")

    @classmethod
    def distance(cls, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        query_embedding = query_embedding.flatten()
        doc_embedding = doc_embedding.flatten()
        query_embedding = query_embedding.reshape(1, -1)
        doc_embedding = doc_embedding.reshape(1, -1)
        return float(cosine_similarity(query_embedding, doc_embedding)[0][0])


@dataclass
class Document:
    filename: str
    embeddings: list[np.ndarray]
    text: str

    @classmethod
    def from_dict(cls, data: dict) -> Document:
        embeddings = [np.array(emb).flatten() for emb in data["embeddings"]]
        return cls(filename=data["filename"], embeddings=embeddings, text=data.get("text", ""))


class Store:
    db: list[Document] = []

    @classmethod
    def load(cls, filepath: str = VECTORSTORE_FILE) -> None:
        assert os.path.exists(filepath), f"Vectorstore file not found at {filepath}. PDF RAG will not work."
        with open(filepath, "r") as f:
            raw_data = json.load(f)

        cls.db = []
        for item in raw_data:
            doc = Document.from_dict(item)
            if not doc.text:
                filepath = os.path.join(PDF_DIR, doc.filename)
                if os.path.exists(filepath):
                    loader = PyPDFLoader(filepath)
                    pages = loader.load()
                    doc.text = "\n".join(page.page_content for page in pages)
            cls.db.append(doc)

        total_embeddings = sum(len(doc.embeddings) for doc in cls.db)
        Out.green(f"Store loaded with {len(cls.db)} documents and {total_embeddings} total embeddings")

    @classmethod
    def search(cls, query_embedding: np.ndarray, n: int = 3) -> list[tuple[Document, float]]:
        assert cls.db, "Store.db has not been initialized."
        query_embedding = query_embedding.flatten()

        all_similarities: list[tuple[Document, float]] = []
        for doc in cls.db:
            for emb in doc.embeddings:
                similarity = Vector.distance(query_embedding, emb)
                all_similarities.append((doc, similarity))

        all_similarities.sort(key=lambda x: x[1], reverse=True)

        top_n_unique: list[tuple[Document, float]] = []
        seen_filenames = set()

        for doc, score in all_similarities:
            if doc.filename not in seen_filenames:
                seen_filenames.add(doc.filename)
                top_n_unique.append((doc, score))
                if len(top_n_unique) == n:
                    break

        Out.blue(f"Top {len(top_n_unique)} similarities: {[score for _, score in top_n_unique]}")
        Out.blue(f"Top {len(top_n_unique)} documents: {[doc.filename for doc, _ in top_n_unique]}")
        return top_n_unique


class Brain:
    model: ChatOpenAI | None = None

    @classmethod
    def load(cls) -> None:
        cls.model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
        Out.green(f"Brain model loaded with {cls.model.model_name}")


class Template:
    RAG = PromptTemplate(
        input_variables=["input", "context"],
        template=(
            "You are an assistant specializing in analyzing medical consultation PDFs. "
            "For each relevant medical case found in the PDFs, provide a comprehensive analysis including:\n"
            "1. The specific PDF documents that contain relevant information\n"
            "2. For each case found:\n"
            "   - The diagnosis given\n"
            "   - The treatment or medication prescribed\n"
            "   - The doctor's specific recommendations\n"
            "3. A summary comparing the cases if multiple relevant documents are found\n\n"
            "Important guidelines:\n"
            "- Always mention which PDF documents you're referencing\n"
            "- If multiple similar cases exist, compare their treatments and recommendations\n"
            "- If the information requested isn't found in the documents, clearly state this\n"
            "- Include any prescribed medications and their dosages\n"
            "- Emphasize that these are reference cases and the patient should consult a healthcare professional\n\n"
            "Context from PDF documents:\n{context}\n\n"
            "Question from the patient: {input}\n"
            "Your Analysis:"
        ),
    )

    TRANSLATE = PromptTemplate(
        input_variables=["text"],
        template=(
            "Translate the following English medical text to Spanish, using simple and clear language that a non-medical audience can understand. "
            "If there are medical terms, provide simple explanations in parentheses. Keep the tone friendly and accessible.\n\n"
            "English text: {text}\n\n"
            "Simple Spanish translation:"
        ),
    )

    SPLIT_PROBLEMS = PromptTemplate(
        input_variables=["symptoms"],
        template=(
            "Split the following patient symptoms/problems into distinct medical conditions that should be analyzed separately. "
            "For each condition, provide:\n"
            "1. The main symptom or condition\n"
            "2. Related symptoms that might be connected\n"
            "3. A search query to find relevant medical cases\n\n"
            "Format the response as a JSON array where each object has the fields: "
            "'condition', 'related_symptoms', and 'search_query'.\n\n"
            "Patient symptoms: {symptoms}\n\n"
            "Split conditions:"
        ),
    )


class Conversation:
    @classmethod
    def load(cls, llm: ChatOpenAI, cache_path: str = CONVERSATION_CACHE, max_token_limit: int = 1000) -> ConversationSummaryBufferMemory:
        Out.blue(f"Loading conversation history from {cache_path}")
        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_token_limit, memory_key="chat_history", return_messages=True)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                memory.chat_memory = pickle.load(f)
        return memory

    @classmethod
    def save(cls, memory: ConversationSummaryBufferMemory, cache_path: str = CONVERSATION_CACHE) -> None:
        Out.blue(f"Saving conversation history to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(memory.chat_memory, f)
        Out.green("Conversation history saved successfully")


class PDFVectorRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> list[LC_Document]:
        query_embedding = Vector.model.encode(query)
        top_docs = Store.search(query_embedding, n=3)
        docs = []
        for doc, _ in top_docs:
            filepath = os.path.join(PDF_DIR, doc.filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            for page in pages:
                docs.append(LC_Document(page_content=page.page_content, metadata={"source": doc.filename}))
        return docs

    async def aget_relevant_documents(self, query: str) -> list[LC_Document]:
        return self.get_relevant_documents(query)


class Agent:
    EXIT_COMMANDS: set[str] = {"salir", "exit", "quit", "bye", "goodbye", "adiós", "chau"}

    def __init__(self):
        assert Brain.model, "Brain not initialized. Cannot create agent."
        self.memory: ConversationSummaryBufferMemory = Conversation.load(Brain.model)

        self.retriever = PDFVectorRetriever()

        combine_docs_chain = create_stuff_documents_chain(llm=Brain.model, prompt=Template.RAG)

        self.rag_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=combine_docs_chain,
        )

        self.tools: list[Tool] = [
            Tool(name="SearchWeb", func=Tools.search, description=Tools.search.__doc__),
            Tool(name="Recommend", func=self.recommend, description="Answers medical questions using PDF documents with history-aware retrieval"),
            Tool(name="TranslateToSpanish", func=Tools.translate_to_spanish, description=Tools.translate_to_spanish.__doc__),
            Tool(name="SplitProblems", func=Tools.split_medical_problems, description=Tools.split_medical_problems.__doc__),
        ]

        self.executor: AgentExecutor = initialize_agent(
            tools=self.tools,
            llm=Brain.model,
            agent="chat-conversational-react-description",
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": (
                    "You are a bilingual (English-Spanish) medical assistant specializing in analyzing medical consultation PDFs. "
                    "Always respond in the same language as the user's question. "
                    "For English questions, answer in English. For Spanish questions, answer in Spanish. "
                    "When a user presents any medical symptoms or health-related questions:\n"
                    "1. FIRST use the SplitProblems tool to break down complex symptoms into distinct conditions\n"
                    "2. Then use the Recommend tool for EACH condition separately to provide comprehensive analysis including:\n"
                    "   - Similar cases found in the documents\n"
                    "   - Treatments and medications prescribed in those cases\n"
                    "   - Specific recommendations given by doctors\n"
                    "   - Comparisons between different cases if available\n"
                    "3. Finally, provide a unified analysis that considers potential interactions between conditions\n\n"
                    "When answering in Spanish, use simple and clear language that a non-medical audience can understand, "
                    "and include brief explanations in parentheses for medical terms. "
                    "Always emphasize that the information provided is from reference cases and the user should consult "
                    "a healthcare professional for personalized medical advice. "
                    "If the medical information needed is not found in the PDFs, clearly state this "
                    "and suggest consulting a healthcare professional."
                )
            },
        )

    @classmethod
    def load(cls) -> None:
        Brain.load()
        Store.load()
        Vector.load()

    def recommend(self, query: str) -> str:
        inputs = {
            "input": query,
            "chat_history": self.memory.chat_memory.messages[-3:] if self.memory.chat_memory.messages else [],
        }
        return self.rag_chain.invoke(inputs)["answer"]

    def ask(self, query: str) -> str:
        return self.executor.run(input=query)

    def info(self) -> None:
        Out.yellow("LangChain Agent Information")
        Out.white(f"Agent: {self.executor.agent}")
        Out.white(f"Executor: {self.executor}")
        Out.white(f"Retriever: {self.retriever}")
        Out.white(f"PDFs indexed: {len(Store.db)}")
        Out.white(f"Memory: '{self.memory}'")
        Out.cyan("Available Tools:")
        for tool in self.tools:
            description = tool.description if isinstance(tool.description, str) else "No description available."
            Out.white(f"- {tool.name}: {description.strip().splitlines()[0]}")

    def start(self):
        Out.green("\n\n\n¡Bienvenido! Soy su asistente médico virtual. ¿En qué puedo ayudarle hoy?")
        Out.yellow(f"Type one of {', '.join(sorted(self.EXIT_COMMANDS))} to end.")

        while True:
            user_input = Out.input(">>> ")
            if user_input.lower() in self.EXIT_COMMANDS:
                Out.yellow("Agent: Goodbye!")
                break
            if not user_input:
                continue
            response = self.ask(user_input)
            Out.green(f"Agent: {response}")

        Conversation.save(self.memory)
        Out.green("Chat ended.")


class Tools:
    @staticmethod
    def search(query: str) -> str:
        """
        Performs a web search using Google and returns the first 2000 characters of the text content.
        Useful for finding current information or topics not covered in the internal knowledge or documents.
        Input must be a search query string.
        """
        url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        main_content = soup.find("body")
        text = main_content.get_text(separator=" ", strip=True) if main_content else soup.get_text(separator=" ", strip=True)
        return text[:2000] if text else "No content found."

    @staticmethod
    def split_medical_problems(symptoms: str) -> str:
        """
        Splits complex medical symptoms or conditions into distinct problems for separate analysis.
        Returns a JSON array of conditions with related symptoms and search queries.
        Input must be the patient's symptoms or medical problems.
        """
        assert Brain.model, "Brain not initialized. Cannot perform problem splitting."
        chain = LLMChain(llm=Brain.model, prompt=Template.SPLIT_PROBLEMS)
        return chain.run(symptoms=symptoms)

    @staticmethod
    def extract_pdf_text(filename: str) -> str:
        """
        Extracts and returns the text content of a specific PDF file given its filename using PyPDFLoader.
        The filename must exactly match one of the files listed by `list_pdfs`.
        Input must be the filename.
        Returns the full text content of the PDF.
        """
        filepath = os.path.join(PDF_DIR, filename)
        assert os.path.exists(filepath), f"Error: PDF file '{filename}' not found in directory '{PDF_DIR}'."

        loader = PyPDFLoader(filepath)
        docs = loader.load()
        assert docs, f"No content loaded from {filename} using PyPDFLoader."

        text = "\n".join(doc.page_content for doc in docs)
        return " ".join(text.split())

    @staticmethod
    def translate_to_spanish(text: str) -> str:
        """
        Translates English medical text to simple Spanish that non-medical audiences can understand.
        Medical terms will include simple explanations in parentheses.
        Input must be the English text to translate.
        """
        assert Brain.model, "Brain not initialized. Cannot perform translation."
        chain = LLMChain(llm=Brain.model, prompt=Template.TRANSLATE)
        return chain.run(text=text)


if __name__ == "__main__":
    Agent.load()
    agent = Agent()
    agent.info()
    agent.start()
