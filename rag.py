from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain.schema.retriever import BaseRetriever
from dotenv import load_dotenv
import os

load_dotenv()

class Rag:
    def __init__(self, vectorstore: PineconeVectorStore, retriever: BaseRetriever):
        self.model = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.system_prompt = (
           """
            You are "Ask me about RAG", a knowledgeable librarian specializing in RAG research papers. A user has requested assistance with research paper recommendations.

            We have retrieved {num_docs} research paper(s) related to the user's query. These papers are listed below:

            {context}

            Please provide detailed information for EACH research paper retrieved, including:
            1. The title of the research paper.
            2. A concise summary of its content, highlighting key findings or topics covered.
            3. Relevant details for locating or referencing the paper (e.g., a link, DOI, university, journal name, or organization).

            Format your response as a numbered list, preserving the order in which the papers were retrieved.
            """
        )
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.vectorstore = vectorstore
        self.retriever = retriever

    def storeDocumentsInVectorstore(self, documents):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorstore.add_documents(documents=documents, ids=uuids)
    
    def createRagChain(self):
        self.question_answer_chain = create_stuff_documents_chain(self.model, self.qa_prompt)
        self.history_aware_retriever = create_history_aware_retriever(self.model, self.retriever, self.contextualize_q_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

    def generateResponse(self, question, chat_history):
        retrieved_docs = self.vectorstore.as_retriever().get_relevant_documents(question)
        num_docs = len(retrieved_docs)

        ai_msg = self.rag_chain.invoke({
            "num_docs": num_docs,
            "input": question,
            "chat_history": chat_history
        })
        return ai_msg
            
