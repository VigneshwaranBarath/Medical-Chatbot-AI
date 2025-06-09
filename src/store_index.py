from helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

print("Loading environment variables...")

PINECONE_API_KEY = "pcsk_3fw9Cf_82M3f1scBV4xC9AsaLm895SnPQrd21HkvFUZrFnJ8S4agseHXpwogFQ3ukPYGPW"

print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "mediapp"
dimension = 384

# Check if index exists, create if not
if not pc.list_indexes() or index_name not in [idx['name'] for idx in pc.list_indexes()]:
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Pinecone index '{index_name}' already exists.")

# Connect to the index
index = pc.Index(index_name)

# Download the Embeddings from Hugging Face
embedding_model = download_hugging_face_embeddings()

# If the index was just created, add documents
if index.describe_index_stats().total_vector_count == 0:
    print("Loading PDF files from Data/")
    extracted_data = load_pdf_file(data='Data/')
    print(f"Extracted data length: {len(extracted_data)}")

    print("Splitting extracted data into text chunks...")
    text_chunks = text_split(extracted_data)
    print(f"Number of text chunks: {len(text_chunks)}")

    print("Embedding text chunks and upserting into Pinecone index...")
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model,pinecone_api_key=PINECONE_API_KEY)
    from uuid import uuid4
    ids = [str(uuid4()) for _ in range(len(text_chunks))]
    vector_store.add_documents(documents=text_chunks, ids=ids)
    print("Upsert complete.")
else:
    print("Index already contains vectors, skipping upsert.")

# Query the vector store
vector_store = PineconeVectorStore(index=index, embedding=embedding_model,pinecone_api_key=PINECONE_API_KEY)
print("Querying the vector store for a test search...")
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

print("Index loaded and ready.")