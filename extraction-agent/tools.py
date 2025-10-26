from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re
import ast
from langchain.docstore.document import Document
import json

def get_content(metadata: str):
    pattern = r"'report_id':\s*'([^']+)'"
    match = re.search(pattern, metadata)
    report_id = match.group(1)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="patient-report-collection",
        embedding_function=embeddings,
        persist_directory="../document-save-agent/chroma"
    )
    results = vector_store.get(
        where={"report_id": report_id},
        include=["documents", "metadatas"]
    )
    txt = ''
    for i, result in enumerate(results["documents"]):
        txt += result
    print(txt)
    return txt

def save_findings(action_input: dict):
    print(action_input)
    print("--------------------")
    if isinstance(action_input, str):
        action_input = ast.literal_eval(action_input) 
    findings = action_input.get("findings")
    values = action_input.get("values")
    metadata = action_input.get("metadata")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="patient-report-findings",
        embedding_function=embeddings,
        persist_directory="./chroma"
    )
    print(findings)
    print("-------------")
    print(values)
    print("--------------------")
    document = Document(
    page_content=json.dumps(
        {"findings": findings, "values": values},
        ensure_ascii=False
    ),
    metadata=metadata
    )
    print("findings saved")
    vector_store.add_documents(documents=[document])