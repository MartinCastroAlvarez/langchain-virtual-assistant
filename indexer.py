import os
import json
import hashlib
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import glob
from tqdm import tqdm
from dataclasses import dataclass, asdict, field

PDF_DIR = "pdfs"
DATABASE_FILE = "vectorstore.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR = "./.cache/sentence_transformers"


@dataclass
class Document:
    filename: str
    text: str
    hash: str
    embedding: list[float]

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "hash": self.hash,
            "embedding": self.embedding,
        }


@dataclass
class Pdf:
    filepath: str

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

    def get_text(self) -> str | None:
        try:
            doc = fitz.open(self.filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            text = " ".join(text.replace("\n", " ").split())
            return text
        except Exception as e:
            print(f"Error extracting text from {self.filename}: {e}")
            return None


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
        for filepath in tqdm(pdf_filepaths, desc="Reading PDFs and Hashing"):
            pdf = Pdf(filepath)
            file_hash = pdf.get_hash()
            text = pdf.get_text()
            assert text, f"No text could be extracted from {pdf.filename}."
            document = Document(filename=pdf.filename, text=text, hash=file_hash, embedding=[])
            documents.append(document)

        print("Generating embeddings...")
        texts: list[str] = [doc.text for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()

        print("Embedding documents...")
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        print("Writing database...")
        database = [doc.to_dict() for doc in documents]
        with open(self.db_file, "w") as f:
            json.dump(database, f, indent=4)
        print(f"Database successfully created at {self.db_file} with {len(database)} entries.")


if __name__ == "__main__":
    indexer = Indexer(pdf_dir=PDF_DIR, db_file=DATABASE_FILE, model_name=EMBEDDING_MODEL, cache_dir=CACHE_DIR)
    indexer.run()
