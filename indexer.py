import glob
import hashlib
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import fitz
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PDF_DIR = "pdfs"
DATABASE_FILE = "vectorstore.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR = "./.cache/sentence_transformers"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass
class Document:
    filename: str
    chunk_index: int
    text: str
    hash: str
    embedding: list[float]

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "hash": self.hash,
            "embedding": self.embedding,
        }


@dataclass
class Pdf:
    filepath: str
    text_splitter: RecursiveCharacterTextSplitter = field(init=False)

    def __post_init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    @property
    def filename(self) -> str:
        return os.path.basename(self.filepath)

    def get_hash(self) -> str:
        hasher = hashlib.sha256()
        with open(self.filepath, "rb") as file:
            while True:
                chunk = file.read(4096)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_chunks(self) -> list[str]:
        doc = fitz.open(self.filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        text = " ".join(text.replace("\n", " ").split())
        return self.text_splitter.split_text(text)


@dataclass
class Indexer:
    pdf_dir: str
    db_file: str
    model_name: str
    cache_dir: str
    _model: SentenceTransformer | None = field(init=False, default=None)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            print("Model loaded.")
        return self._model

    def run(self):
        assert os.path.exists(self.pdf_dir), f"Error: Directory '{self.pdf_dir}' not found."
        pdf_filepaths = glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
        assert pdf_filepaths, f"No PDF files found in '{self.pdf_dir}'."

        print(f"Processing {len(pdf_filepaths)} PDF files...")
        documents: list[Document] = []
        
        for filepath in tqdm(pdf_filepaths, desc="Reading PDFs and Chunking"):
            pdf = Pdf(filepath)
            file_hash = pdf.get_hash()
            chunks = pdf.get_chunks()
            
            for i, chunk_text in enumerate(chunks):
                document = Document(
                    filename=pdf.filename,
                    chunk_index=i,
                    text=chunk_text,
                    hash=f"{file_hash}_{i}",
                    embedding=[]
                )
                documents.append(document)

        print(f"Generating embeddings for {len(documents)} chunks...")
        texts = [doc.text for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        print("Embedding documents...")
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding.tolist()

        print("Writing database...")
        database = [doc.to_dict() for doc in documents]
        with open(self.db_file, "w") as f:
            json.dump(database, f, indent=4)
        print(f"Database successfully created at {self.db_file} with {len(database)} entries.")


if __name__ == "__main__":
    indexer = Indexer(pdf_dir=PDF_DIR, db_file=DATABASE_FILE, model_name=EMBEDDING_MODEL, cache_dir=CACHE_DIR)
    indexer.run()
