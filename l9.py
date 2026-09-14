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

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")


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

Answer the user's question using  the provided context.
if not provided u can give your own ansers


Context:
{context}"""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        "{question}"
    )
])


# 9. Create empty chat history
chat_history = []


# 10. Create conversational RAG chain
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


# 11. Terminal chatbot loop
print("\n===== RAG CHATBOT =====")
print("Ask questions about your document.")
print("Type 'exit' or 'quit' to stop.\n")


while True:

    # Take question from terminal
    question = input("You: ")

    # Exit condition
    if question.lower() in ["exit", "quit"]:
        print("\nChat ended.")
        break

    # Ignore empty questions
    if not question.strip():
        print("Please enter a question.")
        continue

    # Get answer from RAG chain
    answer = rag_chain.invoke(question)

    # Display answer
    print("\nAI:", answer)

    # Save conversation in memory
    chat_history.append(
        HumanMessage(content=question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    print()