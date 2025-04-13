from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import requests
from bs4 import BeautifulSoup
from colorama import Fore, Style
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in environment variables."

PDF_DIR: str = "pdfs"
VECTORSTORE_FILE: str = "vectorstore.json"
CONVERSATION_CACHE: str = "conversation_cache.pkl"
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
CACHE_DIR: str = "./.cache/sentence_transformers"


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

    @classmethod
    def from_dict(cls, data: dict) -> Document:
        embeddings = [np.array(emb).flatten() for emb in data["embeddings"]]
        return cls(filename=data["filename"], embeddings=embeddings)


class Store:
    db: list[Document] = []

    @classmethod
    def load(cls, filepath: str = VECTORSTORE_FILE) -> None:
        assert os.path.exists(filepath), f"Vectorstore file not found at {filepath}. PDF RAG will not work."
        with open(filepath, "r") as f:
            raw_data = json.load(f)
        cls.db = [Document.from_dict(item) for item in raw_data]
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
        input_variables=["question", "context"],
        template=(
            "You are an assistant specializing in analyzing medical consultation PDFs. "
            "Answer the following question based *only* on the provided context from relevant PDF documents. "
            "If the context doesn't contain the answer, state that the information is not available in the provided documents. "
            "Explicitly mention the filename(s) from the context that support your answer. The context contains markers like '--- Context from filename.pdf ---'.\n\n"
            "Context from PDF documents:\n{context}\n\n"
            "Question from the patient: {question}\n"
            "Your Answer:"
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
    def find_relevant_pdfs(query: str, top_k: int = 10) -> str:
        """
        Finds the most relevant PDF documents based on the user query using embeddings.
        Returns a JSON string list of dictionaries, each containing 'filename' and 'score'.
        Input must be the user's query string.
        """
        assert Store.db, "Store.db has not been initialized."
        assert Vector.model, "Vector model not loaded. Cannot perform PDF search."
        
        query = query.lower().strip()
        query_embedding = Vector.model.encode(query)
        query_embedding = query_embedding.flatten() if query_embedding.ndim > 1 else query_embedding
        
        top_docs = Store.search(query_embedding, top_k)
        results = []
        for doc, score in top_docs:
            Out.blue(f"Document: {doc.filename}, Score: {score}")
            results.append({
                "filename": doc.filename,
                "score": score
            })
        return json.dumps(results)

    @staticmethod
    def recommend(query: str) -> str:
        """
        Answers questions based on the content of available PDF medical consultation documents.
        Uses FindRelevantPDFs and ExtractPDFText to gather context, then synthesizes an answer.
        Input must be the user's question.
        """
        assert Brain.model, "Brain not initialized. Cannot generate recommendations."
        relevant_files_json = Tools.find_relevant_pdfs(query)
        relevant_files_info = json.loads(relevant_files_json)
        assert relevant_files_info, "No relevant PDF documents found for your query."

        context_parts = []
        extracted_filenames = set()

        for file_info in relevant_files_info:
            filename = file_info["filename"]
            doc_text = Tools.extract_pdf_text(filename)
            context_parts.append(f"--- Context from {filename} ---\n{doc_text}\n--- ")
            extracted_filenames.add(filename)

        context = "\n".join(context_parts)
        rag_chain = LLMChain(llm=Brain.model, prompt=Template.RAG)
        response = rag_chain.run(question=query, context=context)
        return response

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


class Agent:
    EXIT_COMMANDS: set[str] = {"salir", "exit", "quit", "bye", "goodbye", "adiós", "chau"}

    def __init__(self):
        assert Brain.model, "Brain not initialized. Cannot create agent."
        self.memory: ConversationSummaryBufferMemory = Conversation.load(Brain.model)

        self.tools: list[Tool] = [
            Tool(name='SearchWeb', func=Tools.search, description=Tools.search.__doc__),
            Tool(name='ExtractPDFText', func=Tools.extract_pdf_text, description=Tools.extract_pdf_text.__doc__),
            Tool(name='FindRelevantPDFs', func=Tools.find_relevant_pdfs, description=Tools.find_relevant_pdfs.__doc__),
            Tool(name='Recommend', func=Tools.recommend, description=Tools.recommend.__doc__),
            Tool(name='TranslateToSpanish', func=Tools.translate_to_spanish, description=Tools.translate_to_spanish.__doc__),
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
                    "When a user presents any medical symptoms or health-related questions, ALWAYS use the Recommend tool first "
                    "to search through the medical PDFs and provide evidence-based information. "
                    "When answering in Spanish, use simple and clear language that a non-medical audience can understand, "
                    "and include brief explanations in parentheses for medical terms. "
                    "Base your answers on the provided context from relevant PDF documents and always reference which documents "
                    "you used in your response. If the medical information needed is not found in the PDFs, clearly state this "
                    "and suggest consulting a healthcare professional."
                )
            },
        )

    def ask(self, query: str) -> str:
        return self.executor.run(input=query)


    def info(self) -> None:
        Out.yellow("LangChain Agent Information")
        Out.white(f"Agent: {self.executor.agent}")
        Out.white(f"Executor: {self.executor}")
        Out.cyan("Available Tools:")
        for tool in self.tools:
            description = tool.description if isinstance(tool.description, str) else "No description available."
            Out.white(f"- {tool.name}: {description.strip().splitlines()[0]}")
        Out.cyan(f"PDFs indexed: {len(Store.db)}")
        Out.cyan(f"Memory: '{self.memory}'")

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


if __name__ == "__main__":
    Brain.load()
    Store.load()
    Vector.load()
    agent = Agent()
    agent.info()
    agent.start()
