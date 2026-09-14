import os

import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Conversational RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Conversational RAG Chatbot")
st.caption("Ask questions about the information stored in data.txt")


# --------------------------------------------------
# 2. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is missing in the .env file.")
    st.stop()


# --------------------------------------------------
# 3. Load and prepare the document
# --------------------------------------------------

@st.cache_resource
def create_vector_store():

    with open("data.txt", "r", encoding="utf-8") as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    return vector_store


# --------------------------------------------------
# 4. Create vector store and retriever
# --------------------------------------------------

vector_store = create_vector_store()

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# --------------------------------------------------
# 5. Create Gemini model
# --------------------------------------------------

@st.cache_resource
def create_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )


llm = create_llm()


# --------------------------------------------------
# 6. Create question-rewriting prompt
# --------------------------------------------------

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given the chat history and the latest user question,
rewrite the latest question as a complete standalone question.

Do not answer the question.
Only return the rewritten question."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


# --------------------------------------------------
# 7. Create final-answer prompt
# --------------------------------------------------

answer_prompt = ChatPromptTemplate.from_messages([
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
    ("human", "{question}")
])


# --------------------------------------------------
# 8. Create question-rewriting chain
# --------------------------------------------------

rewrite_chain = (
    rewrite_prompt
    | llm
    | StrOutputParser()
)


# --------------------------------------------------
# 9. Conversational RAG function
# --------------------------------------------------

def conversational_rag(question, chat_history):

    # Step 1: Rewrite the question using previous messages
    standalone_question = rewrite_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    # Step 2: Retrieve relevant documents
    documents = retriever.invoke(standalone_question)

    # Step 3: Combine retrieved chunks into context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Step 4: Create the final prompt
    final_prompt = answer_prompt.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })

    # Step 5: Ask Gemini to generate the answer
    response = llm.invoke(final_prompt)

    return response.content


# --------------------------------------------------
# 10. Initialize Streamlit session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# 11. Clear chat button
# --------------------------------------------------

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []
    st.session_state.chat_history = []

    st.rerun()


# --------------------------------------------------
# 12. Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 13. Chat input
# --------------------------------------------------

question = st.chat_input("Ask a question about the document...")


if question:

    # Display user's question immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Save user's question for display
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = conversational_rag(
                question,
                st.session_state.chat_history
            )

        st.markdown(answer)

    # Save AI answer for display
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Save conversation history for LangChain
    st.session_state.chat_history.append(
        HumanMessage(content=question)
    )

    st.session_state.chat_history.append(
        AIMessage(content=answer)
    )