import glob
import json
import os
import re
from dataclasses import dataclass

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

BOILERPLATE_PATTERNS = [
    r"Informe de Consulta Médica",
    r"Fecha de consulta: \d{2}/\d{2}/\d{4}",
    r"Paciente: [A-Za-zÁ-Úá-úÑñ\s]+",
    r"Médico: Dr\. [A-Za-zÁ-Úá-úÑñ\s]+",
    r"Motivo de la consulta:",
    r"Diagnóstico:",
    r"Recomendación:",
    r"Medicamento prescripto:",
]

COMMON_WORDS = {
    "fecha",
    "consulta",
    "paciente",
    "médico",
    "motivo",
    "diagnóstico",
    "recomendación",
    "medicamento",
    "prescripto",
    "dr",
    "doctor",
    "dra",
    "doctora",
    "informe",
    "médica",
    "medical",
    "consultation",
    "report",
    "día",
    "mes",
    "año",
    "tratamiento",
    "análisis",
    "evaluación",
    "realizar",
    "iniciar",
    "programar",
    "controlar",
}

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)


@dataclass
class Document:
    filename: str
    text: str
    embeddings: list[list[float]]

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "embeddings": self.embeddings,
        }


@dataclass
class Pdf:
    filepath: str

    @property
    def filename(self) -> str:
        return os.path.basename(self.filepath)

    def clean(self, text: str) -> str:
        cleaned = text.lower()
        for pattern in BOILERPLATE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        words = cleaned.split()
        return " ".join(word for word in words if word.lower() not in COMMON_WORDS)

    def split(self) -> list[str]:
        doc = fitz.open(self.filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        text = self.clean(" ".join(text.split()))
        chunks = TEXT_SPLITTER.split_text(text)
        return [chunk for chunk in chunks if len(chunk.split()) > 5]


@dataclass
class Indexer:
    pdf_dir: str
    db_file: str
    model_name: str
    cache_dir: str
    _model: SentenceTransformer | None = None

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
        documents: list[Document] = []
        for i, filepath in enumerate(pdf_filepaths, 1):
            print(f"Document {i}/{len(pdf_filepaths)}")
            pdf = Pdf(filepath)
            document = Document(filename=pdf.filename, text="", embeddings=[])
            chunks = pdf.split()
            if chunks:
                document.text = chunks[0]
                chunk_embeddings = self.model.encode(chunks, convert_to_numpy=True)
                document.embeddings = [embedding.tolist() for embedding in chunk_embeddings]
                documents.append(document)

        print(f"Processed {len(documents)} documents")
        print("Writing database...")
        database = [doc.to_dict() for doc in documents]
        with open(self.db_file, "w") as f:
            json.dump(database, f, indent=4)

        print(f"Database successfully created at {self.db_file}")


if __name__ == "__main__":
    indexer = Indexer(pdf_dir=PDF_DIR, db_file=DATABASE_FILE, model_name=EMBEDDING_MODEL, cache_dir=CACHE_DIR)
    indexer.run()
