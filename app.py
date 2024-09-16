from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
import time
import gradio as gr
import requests
from typing import Generator
    
chat_history = []

def generate_response(chat_input: str, bot_message: str) -> Generator[str, str, str] | str:
    url = "http://127.0.0.1:8000/generatechat/"
    payload = {
        'question': chat_input,
    }
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        answer = data['response']['answer']
        print("Success:", response.json())
        
        # Get a typewriting animation response
        partial_response = ""
        for char in answer:
            partial_response += char
            yield partial_response
            time.sleep(0.005)
    else:
        print("Error:", response.status_code, response.text)
        return f"Error: {response.status_code}, {response.text}"

CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    chatbot = gr.ChatInterface(
        fn=generate_response, 
        title="AskmeAboutRAG Chat",
        description="RAG model for asking about RAG",
        chatbot=chatbot,
    )

if __name__ == "__main__":
    demo.launch()