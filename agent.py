from __future__ import annotations
import os
import json
import pickle
import requests
import numpy as np
from dataclasses import dataclass
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in environment variables."

PDF_DIR = "pdfs"
VECTORSTORE_FILE = "vectorstore.json"
CONVERSATION_CACHE = "conversation_cache.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = "./.cache/sentence_transformers"
VECTOR_DATA = {}
QUERY_EMBEDDING_MODEL = None


@dataclass
class Document:
    filename: str
    hash: str
    embedding: np.ndarray

    @classmethod
    def from_dict(cls, data: dict) -> Document:
        return cls(filename=data["filename"], hash=data["hash"], embedding=np.array(data["embedding"]))


class Store:
    @classmethod
    def load(cls, filepath: str) -> list[Document]:
        if not os.path.exists(filepath):
            print(f"Warning: Vectorstore file not found at {filepath}. PDF RAG will not work.")
            return []
        with open(filepath, "r") as f:
            raw_data = json.load(f)
        documents = [Document.from_dict(item) for item in raw_data]
        print(f"Loaded {len(documents)} documents from {filepath}")
        return documents


class Template:
    CONTEXT = PromptTemplate(
        input_variables=[],
        template=(
            "You are an assistant specializing in analyzing medical consultation PDFs. "
            "Answer the following question based *only* on the provided context from relevant PDF documents. "
        ),
    )
    RAG_PROMPT = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "If the context doesn't contain the answer, state that the information is not available in the provided documents. "
            "Explicitly mention the filename(s) from the context that support your answer. The context contains markers like '--- Context from filename.pdf ---'.\n\n"
            "Context from PDF documents:\n{context}\n\n"
            "Question from the patient: {question}\n"
            "Your Answer:"
        ),
    )


class Conversation:
    @classmethod
    def load(cls, llm: ChatOpenAI, cache_path: str, max_token_limit: int = 1000) -> ConversationSummaryBufferMemory:
        print(f"Loading conversation history from {cache_path}")
        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_token_limit, memory_key="chat_history", return_messages=True)
        if os.path.exists(cache_path):
            print(f"Loading conversation history from {cache_path}")
            with open(cache_path, "rb") as f:
                memory.chat_memory = pickle.load(f)
        return memory

    @classmethod
    def save(cls, memory: ConversationSummaryBufferMemory, cache_path: str) -> None:
        print(f"Saving conversation history to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(memory.chat_memory, f)
        print(f"Conversation history saved to {cache_path}")


class AgentTools:

    @staticmethod
    def search(query: str) -> str:
        """
        Performs a web search using Google and returns the first 2000 characters of the text content.
        Useful for finding current information or topics not covered in the internal knowledge or documents.
        Input must be a search query string.
        """
        print(f"Searching for {query} on Google...")
        try:
            url = f"https://www.google.com/search?q={query}"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text()
            return text[:2000] if text else "No content found."
        except requests.RequestException as e:
            return f"Error during web search: {e}"
        except Exception as e:
            return f"An unexpected error occurred during web search: {e}"

    @staticmethod
    def list_pdfs(_: str) -> str:
        """
        Lists the filenames of all the PDF documents that are available in the internal knowledge base.
        Does not take any input arguments (input can be an empty string or ignored).
        Returns a newline-separated list of filenames.
        """
        assert VECTOR_DATA, "No PDF documents have been indexed."
        print(f"Listing {len(VECTOR_DATA)} PDF documents...")
        filenames = [doc.filename for doc in VECTOR_DATA]
        return "\n".join(filenames) if filenames else "No PDF filenames found in the index."

    @staticmethod
    def extract_pdf_text(filename: str) -> str:
        """
        Extracts and returns the full text content of a specific PDF file given its filename using PyPDFLoader.
        The filename must exactly match one of the files listed by `list_pdfs`.
        Input must be the exact filename of the PDF.
        """
        print(f"Extracting text from {filename}...")
        filepath = os.path.join(PDF_DIR, filename)
        assert os.path.exists(filepath), f"Error: PDF file '{filename}' not found in directory '{PDF_DIR}'."
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            cleaned_text = " ".join(text.replace("\n", " ").split())
            return cleaned_text if cleaned_text else f"No text could be extracted from {filename}."
        except Exception as e:
            return f"Error extracting text from {filename} using PyPDFLoader: {e}"

    @staticmethod
    def find_relevant_pdfs(query: str, top_k: int = 3) -> str:
        """
        Finds the most relevant PDF documents based on the user query using embeddings.
        Returns a list of filenames and their relevance scores.
        Input must be the user's query string.
        """
        assert VECTOR_DATA, "No vector data loaded. Cannot perform PDF search."
        assert QUERY_EMBEDDING_MODEL, "Query embedding model not loaded. Cannot perform PDF search."

        print(f"Finding relevant PDFs for query: {query}")
        query_embedding = QUERY_EMBEDDING_MODEL.encode([query])
        doc_embeddings = np.array([doc.embedding for doc in VECTOR_DATA])
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        k = min(top_k, len(VECTOR_DATA))
        if k == 0:
            return "No documents available to search."

        top_k_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        print(f"\n--- Found {len(top_k_indices)} relevant documents ---")
        for i in top_k_indices:
            doc_info = VECTOR_DATA[i]
            filename = doc_info.filename
            similarity_score = similarities[i]
            print(f"- {filename} (Similarity: {similarity_score:.4f})")
            results.append({"filename": filename, "score": float(similarity_score)})
        print("-------------------------------------")

        if not results:
            return "No relevant PDF documents found."

        return json.dumps(results)


class Agent:
    def __init__(self, api_key: str, conv_cache_path: str):
        self.api_key = api_key
        self.conv_cache_path = conv_cache_path
        self.llm = ChatOpenAI(temperature=0, openai_api_key=self.api_key, model_name="gpt-3.5-turbo")
        self.memory = Conversation.load(self.llm, self.conv_cache_path)
        self.rag_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(input_variables=["question", "context"], template=Template.CONTEXT.template + "\n" + Template.RAG_PROMPT.template),
        )

        self.tools = [
            Tool(name="SearchWeb", func=AgentTools.search, description=AgentTools.search.__doc__),
            Tool(name="ListAvailablePDFs", func=AgentTools.list_pdfs, description=AgentTools.list_pdfs.__doc__),
            Tool(name="ExtractPDFText", func=AgentTools.extract_pdf_text, description=AgentTools.extract_pdf_text.__doc__),
            Tool(name="FindRelevantPDFs", func=AgentTools.find_relevant_pdfs, description=AgentTools.find_relevant_pdfs.__doc__),
            Tool(
                name="AnswerQuestionFromPDFs",
                func=self._answer_from_pdfs,
                description=(
                    "Answers questions based *specifically* on the content of the available PDF medical consultation documents found using FindRelevantPDFs and ExtractPDFText. "
                    "Use this when the user asks about past cases, diagnoses, recommendations, or specific information likely contained within PDF files. "
                    "Input must be the user's question. This tool will automatically find relevant PDFs, extract their content, and synthesize an answer."
                ),
            ),
        ]

        agent_kwargs = {"system_message": Template.CONTEXT.template}

        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="chat-conversational-react-description",
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            agent_kwargs=agent_kwargs,
        )

    def _answer_from_pdfs(self, query: str) -> str:
        relevant_files_json = AgentTools.find_relevant_pdfs(query)
        if (
            relevant_files_json.startswith("Error:")
            or "No documents available" in relevant_files_json
            or "No relevant PDF documents found" in relevant_files_json
        ):
            return f"Could not find relevant PDF documents for your query. {relevant_files_json}"

        try:
            relevant_files_info = json.loads(relevant_files_json)
            if not relevant_files_info:
                return "No relevant PDF documents found."
        except json.JSONDecodeError:
            return f"Error decoding relevant PDF information: {relevant_files_json}"

        context_parts = []
        extracted_filenames = []
        for file_info in relevant_files_info:
            filename = file_info.get("filename")
            if not filename:
                continue
            print(f"Extracting text from relevant file: {filename}")
            doc_text = AgentTools.extract_pdf_text(filename)
            if not doc_text.startswith("Error:"):
                context_parts.append(f"--- Context from {filename} ---\n{doc_text}\n---")
                extracted_filenames.append(filename)
            else:
                print(f"Warning: {doc_text}")
                context_parts.append(f"--- Could not extract text from {filename} ---")

        if not context_parts:
            return "Could not extract text from any of the relevant PDF documents."

        context = "\n".join(context_parts)

        try:
            response = self.rag_chain.run(question=query, context=context)
            return response
        except Exception as e:
            print(f"Error running RAG chain: {e}")
            return "Error generating answer from PDF context."

    def ask(self, query: str) -> str:
        print(f"\nUser query: {query}")
        try:
            response = self.agent_executor.run(query)
            print(f"Agent response: {response}")
            return response
        except Exception as e:
            print(f"Error during agent execution: {e}")
            Conversation.save(self.memory, self.conv_cache_path)
            return "Sorry, an error occurred while processing your request."

    def start(self):
        print("\n--- LangChain Agent Initialized ---")
        print("Available Tools:")
        for tool in self.tools:
            print(f"- {tool.name}: {tool.description.strip().splitlines()[0]}")
        print(f"PDFs indexed: {len(VECTOR_DATA)}")
        print(f"Conversation cache: '{self.conv_cache_path}'")
        print("Type 'salir', 'exit', or 'quit' to end.")
        print("-----------------------------------")

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["salir", "exit", "quit"]:
                    print("Exiting agent...")
                    break
                if not user_input:
                    continue

                assistant_response = self.ask(user_input)

            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"An unexpected error occurred in the chat loop: {e}")
                break

        Conversation.save(self.memory, self.conv_cache_path)
        print("Chat ended.")


if __name__ == "__main__":
    VECTOR_DATA = Store.load(VECTORSTORE_FILE)
    QUERY_EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=CACHE_DIR)

    agent_instance = Agent(api_key=OPENAI_API_KEY, conv_cache_path=CONVERSATION_CACHE)
    agent_instance.start()
