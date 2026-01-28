# src/ingest_books.py

import os
from sklearn.feature_extraction.text import TfidfVectorizer

def ingest_and_index_books(books_path, chunk_size=800, overlap=400):
    """
    Ingest novels from books_path, chunk text, and build a TF-IDF vector store.
    Returns a dict with 'chunks', 'tfidf_matrix', and 'vectorizer'.
    """

    # -------------------------------
    # Step 1: Read all books
    # -------------------------------
    books_list = []
    for filename in os.listdir(books_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(books_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            books_list.append({
                "book_name": filename.replace(".txt", ""),
                "text": text
            })

    # -------------------------------
    # Step 2: Chunk text with overlap
    # -------------------------------
    chunked_books = []
    for book in books_list:
        words = book["text"].split()
        start = 0
        chunk_id = 0

        while start < len(words):
            chunk_words = words[start:start + chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunked_books.append({
                    "book_name": book["book_name"],
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text
                })
            start += chunk_size - overlap
            chunk_id += 1

    # -------------------------------
    # Step 3: Build TF-IDF vectors
    # -------------------------------
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(
        [chunk["chunk_text"] for chunk in chunked_books]
    )

    # -------------------------------
    # Step 4: Return vector store
    # -------------------------------
    vector_store = {
        "chunks": chunked_books,
        "tfidf_matrix": tfidf_matrix,
        "vectorizer": vectorizer
    }

    return vector_store
