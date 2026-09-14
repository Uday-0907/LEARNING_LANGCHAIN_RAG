
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS


# 1. Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# 2. Read document
with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 3. Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)


# 4. Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


# 5. Store vectors in FAISS
vector_store = FAISS.from_texts(
    chunks,
    embedding=embeddings
)


# 6. User question
question = "What is RAG?"


# 7. Retrieve relevant chunks
results = vector_store.similarity_search(
    question,
    k=2
)


# 8. Combine retrieved chunks
context = "\n".join(
    [result.page_content for result in results]
)

# 9. Create prompt
prompt = """
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""




# 10. Fill the prompt with actual context and question
final_prompt = prompt.format(
    context=context,
    question=question
)


# 11. Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# 12. Send final prompt to Gemini
response = llm.invoke(final_prompt)


# 13. Print retrieved context
print("\nRetrieved Context:")
print(context)


# 14. Print final answer
print("\nFinal Answer:")
print(response.content)


