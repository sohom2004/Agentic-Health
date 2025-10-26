from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import re
from datetime import datetime
from uuid import uuid4
from sentence_transformers import SentenceTransformer

import re
from datetime import datetime

def extract_report_data(text):
    match = (
        re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        or re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", text)
    )
    report_date = None
    if match:
        rep_date = match.group(1)
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
            try:
                report_date = datetime.strptime(rep_date, fmt).date().isoformat()
                break
            except ValueError:
                continue
    if not report_date:
        report_date = datetime.now().date().isoformat()

    conf_match = re.search(r"'confidence'\s*:\s*([0-9]*\.?[0-9]+)", text)
    confidence = float(conf_match.group(1)) if conf_match else None

    return report_date, confidence

def get_next_report_id():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="patient-report-collection",
        embedding_function=embeddings,
        persist_directory="./chroma"
    )
    results = vector_store.get(include=["metadatas"])
    
    if not results["metadatas"]:
        return "RPT-1"
    
    report_ids = [m["report_id"] for m in results["metadatas"] if "report_id" in m]
    max_id = max(int(r.split("-")[1]) for r in report_ids)
    
    return f"RPT-{max_id + 1}"

def convert_text_to_document(report_date, confidence, content, patient_id="pt-1"):
    report_id = get_next_report_id()
    metadata = {
        "report_id": report_id,
        "patient_id": patient_id,
        "report_date": report_date,
        "confidence": confidence,
    }
    document = Document(
    page_content=content,
    metadata=metadata
    )
    return document, metadata

def split_document(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=True,
    )
    chunks = text_splitter.split_documents(document)
    return chunks

def store_in_chroma(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="patient-report-collection",
        embedding_function=embeddings,
        persist_directory="./chroma"
    )
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks)
    print("Documents uploaded and persisted to ChromaDB.")

def store_content(text):
    report_date, confidence= extract_report_data(text)
    document, metadata = convert_text_to_document(report_date, confidence, text)
    chunks = split_document([document])
    store_in_chroma(chunks)
    return metadata