import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded:", bool(api_key))

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=api_key
)

response = model.invoke("What is RAG? Explain in simple words.")

print(response.content[0]["text"])