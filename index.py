import os
import json
import hashlib
import fitz
from sentence_transformers import SentenceTransformer
import glob
from tqdm import tqdm

PDF_DIR = "pdfs"
DATABASE_FILE = "vectorstore.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file:
        while True:
            chunk = file.read(4096)  # Read in chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_text_from_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        # Basic cleaning: replace multiple newlines/spaces
        text = " ".join(text.replace("\n", " ").split())
        return text
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def generate_embeddings(text_list, model):
    """Generates embeddings for a list of texts using the specified model."""
    return model.encode(text_list, show_progress_bar=False).tolist()  # Convert numpy arrays to lists for JSON serialization


def create_database():
    """Reads PDFs, generates hashes and embeddings, and saves to a JSON file."""
    if not os.path.exists(PDF_DIR):
        print(f"Error: Directory '{PDF_DIR}' not found.")
        return

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIR}'.")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    # Load the sentence transformer model
    # Use cache_folder to potentially speed up subsequent runs
    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder="./.cache/sentence_transformers")
    print("Model loaded.")

    database = []
    texts_to_embed = []
    file_info = []

    print(f"Processing {len(pdf_files)} PDF files...")
    # First pass: Extract text and calculate hash
    for pdf_path in tqdm(pdf_files, desc="Reading PDFs and Hashing"):
        filename = os.path.basename(pdf_path)
        file_hash = get_file_hash(pdf_path)
        text = extract_text_from_pdf(pdf_path)

        if text:  # Only process if text extraction was successful
            texts_to_embed.append(text)
            file_info.append({"filename": filename, "hash": file_hash})
        else:
            print(f"Skipping {filename} due to text extraction error.")

    if not texts_to_embed:
        print("No text could be extracted from any PDF files. Database not created.")
        return

    # Second pass: Generate embeddings in batch
    print("Generating embeddings (this may take a while depending on the number of PDFs)...")
    embeddings = generate_embeddings(texts_to_embed, model)

    # Combine info and embeddings
    for i, info in enumerate(file_info):
        database.append({"filename": info["filename"], "hash": info["hash"], "embedding": embeddings[i]})  # Add the corresponding embedding

    # Save the database to a JSON file
    try:
        with open(DATABASE_FILE, "w") as f:
            json.dump(database, f, indent=4)
        print(f"Database successfully created at {DATABASE_FILE}")
    except Exception as e:
        print(f"Error writing database file: {e}")


if __name__ == "__main__":
    create_database()
