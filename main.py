from langchain_core.messages import AIMessage, HumanMessage
from fastapi import FastAPI
from langchain_pinecone.vectorstores import Pinecone
from pydantic import BaseModel
from rag import Rag
from retriever import AskMeAboutRagRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('PINECONE_KEY')
index_name="askmeaboutrag" 

vectorstore = Pinecone(pinecone_api_key=api_key, index_name=index_name, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
retriever = AskMeAboutRagRetriever(vectorstore)
rag_llm = Rag(vectorstore, retriever);

rag_llm.createRagChain()

chat_history = []

class ChatInput(BaseModel):
    question: str

app = FastAPI() 

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generatechat/")
async def generateResponse(chat_input: ChatInput):
    ai_msg = rag_llm.generateResponse(chat_input.question, chat_history)
    chat_history.extend(
        [
            HumanMessage(content=chat_input.question),
            AIMessage(content=ai_msg["answer"]),
        ]
    )
    return {"response": ai_msg}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    print("Server is running")