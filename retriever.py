from langchain.schema.retriever import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_pinecone.vectorstores import Pinecone
from langchain.schema import Document
from pydantic import PrivateAttr

class AskMeAboutRagRetriever(BaseRetriever):
    vectorstore: Pinecone = PrivateAttr()

    def __init__(self, vectorstore: Pinecone, **data):
        super().__init__(**data)
        self.vectorstore = vectorstore
        
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        retrieved_docs = self.vectorstore.as_retriever().get_relevant_documents(query)
        docs = [
            Document(
                page_content= str(i+1) + ".)" + "Title = " + "(" + doc.metadata.get('title') + ")" + " " + "Content = " + "(" + doc.page_content + ")",
                metadata={"title": doc.metadata.get('title')}
            )
            for i, doc in enumerate(retrieved_docs)
        ]
        return docs