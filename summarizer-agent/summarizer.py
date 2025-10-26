from langchain.agents import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from tools import get_all_findings

load_dotenv()

def create_summarizer_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["latest", "history"],
        template="""
    You are a medical summarization agent.

    Task:
    1. Use the latest report to build the main summary.
    2. Use past reports as historical context.
    3. Highlight new changes, worsening symptoms, or improvements.

    Latest findings and values:
    {latest}

    Past reports (for context):
    {history}

    Return JSON ONLY in this format exactly:
    {{
    "summary": "...",
    "key_changes": "...",
    "current_values": {{...}}
    }}
    """
        )

    summarizer_chain = prompt | llm
    return summarizer_chain

def summarize_latest_for_user(user_id):
    reports = get_all_findings(user_id)

    if not reports:
        return "No stored findings for this user."

    latest = reports[-1]
    history = reports[:-1]

    summarizer = create_summarizer_agent()
    response = summarizer.invoke({
    "latest": latest,
    "history": history
    })
    return response

if __name__ == "__main__":
    response = summarize_latest_for_user("pt-1")
    print("\n=== Sumaarizer RESULT ===")
    print(response)
