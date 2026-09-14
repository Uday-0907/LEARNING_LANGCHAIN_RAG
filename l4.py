import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# # 1. Read document
with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# # 2. Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = splitter.split_text(text)

# # 3. Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# # 4. Store chunks as vectors in FAISS
vector_store = FAISS.from_texts(
    chunks,
    embedding=embeddings
)

# # 5. User question
question = "What is RAG?"

# # 6. Search for similar chunks
results = vector_store.similarity_search(
    question,
    k=2
)

# # 7. Print results
print("\nUser Question:")
print(question)

print("\nRetrieved Chunks:")

for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(result.page_content)

