import ollama
import chromadb

# Flow -1: Create embeddings using Ollama and add to ChromaDB

def ollama_embed(texts):
    embeddings = []

    for t in texts:
        res = ollama.embeddings(
            model="nomic-embed-text:latest",
            prompt=t
        )
        embeddings.append(res["embedding"])

    return embeddings

docs =[
    "Python is a programming language",
    "FastAPI is great for building APIs",
    "ChromaDB is a vector database",
    "I love machine learning",
    "PostgreSQL is a relational Database",
]

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="ollama_demo_cosine"
    ,metadata={"hnsw:space": "cosine"}
    )

#Create embeddings using ollama
embeddings = ollama_embed(docs)

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(docs))],
)

print("Added using Ollama Embeddings")

#Flow -2: Search using Ollama embeddings

# query = "What is a good database for vector search?"
query = "Tell me about postgresql"

q_emb = ollama_embed([query])

results = collection.query(
    query_embeddings=q_emb,
    n_results=3,
    include=["documents", "distances"]
)

print("Search results:")
print(f"{results['documents']} \n Distances: {results['distances']}")