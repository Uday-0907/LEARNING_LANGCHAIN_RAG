import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
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


# 7. Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# 8. Create conversational prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful RAG assistant.

Answer the user's question using only the provided context.
Use the chat history to understand follow-up questions.
If the answer is not available in the context, say:
"I don't know based on the provided document."

Context:
{context}"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    (
        "human",
        "{question}"
    )
])


# 9. Create conversational RAG chain
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "chat_history": lambda x: chat_history
    }
    | prompt
    | llm
    | StrOutputParser()
)


# 10. Create empty chat history
chat_history = []


# 11. First question
question = "What is RAG?"

answer = rag_chain.invoke(question)

print("\nAI:", answer)


# 12. Save first conversation
chat_history.append(
    HumanMessage(content=question)
)

chat_history.append(
    AIMessage(content=answer)
)


# 13. Second question
question = "Why is it useful?"

answer = rag_chain.invoke(question)

print("\nAI:", answer)


# 14. Save second conversation
chat_history.append(
    HumanMessage(content=question)
)

chat_history.append(
    AIMessage(content=answer)
)