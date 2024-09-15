import fitz
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pc = Pinecone(api_key=os.get('PINECONE_KEY'))
index_name = "askmeaboutrag" 
index = pc.Index(index_name)

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_pages_from_pdf(pdf_path):
    doc = fitz.open(pdf_path) 
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num) 
        text = page.get_text("text") 
        pages.append(text)
    return pages

def store_document_in_pinecone(document_id, pages, title, model):
    for page_number, page_text in enumerate(pages):
        embedding = model.encode(page_text) 
        index.upsert(
            vectors=[
                {
                    "id": f'{document_id}_page_{page_number}',
                    "values": embedding,
                    "metadata": {
                        "document_id": document_id,
                        "page_number": page_number,
                        "text": page_text,
                        "title": title,
                    }
                }
            ],
        )
    print(f"Stored {len(pages)} pages for document: {document_id}")

def process_pdfs_in_folder(folder_path):
    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            document_id = str(i+1)
            print(f"Processing {filename} with document_id: {document_id}")
            
            pages = extract_pages_from_pdf(pdf_path)
            file_name_without_extension = os.path.splitext(filename)[0]

            store_document_in_pinecone(document_id, pages, file_name_without_extension, model)
            print("Stored Completed")

folder_path = 'files' 
process_pdfs_in_folder(folder_path)
