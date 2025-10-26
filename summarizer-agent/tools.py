from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re
import ast
from langchain.docstore.document import Document
import json

def get_all_findings(user_id):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="patient-report-findings",
        embedding_function=embeddings,
        persist_directory="../extraction-agent/chroma"
    )
    results = vector_store.get(
        where={"patient_id": user_id},
        include=["documents", "metadatas"]
    )
    docs = []
    for doc in results.get("documents", []):
        try:
            docs.append(json.loads(doc))
        except Exception:
            continue
    return docs

def get_recent(user_id="pt-1"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="patient-report-findings",
        embedding_function=embeddings,
        persist_directory="../extraction-agent/chroma"
    )
    results = vector_store.get(
        where={"patient_id": user_id},
        include=["documents", "metadatas"]
    )
    return results.get("documents")[-1]