import faiss
from sentence_transformers import SentenceTransformer



EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

embedding_dim = 384  # output dimension
FAISS_INDEX = faiss.IndexFlatIP(embedding_dim)  # inner product = cosine similarity if normalized

# Store metadata alongside vectors
DOC_STORE = []  # each entry: {"content": chunk, "source": source}