import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 1. Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


# 2. Read document
with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 3. Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)


# 4. Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


# 5. Create FAISS vector store
vector_store = FAISS.from_texts(
    chunks,
    embedding=embeddings
)


# 6. Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# 7. Create prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
""")


# 8. Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# 9. Create RAG chain
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# 10. Ask question
question = "What is RAG?"

response = rag_chain.invoke(question)

print("\nFinal Answer:")
print(response)