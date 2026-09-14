import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# Read document
with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))


# Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# Convert first chunk into vector
vector = embeddings.embed_query(chunks[0])

print("\nFirst Chunk:")
print(chunks[0])

print("\nFirst 10 Vector Values:")
print(vector[:10])

print("\nTotal Vector Dimensions:")
print(len(vector))




