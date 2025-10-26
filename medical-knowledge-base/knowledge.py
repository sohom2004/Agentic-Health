from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

tools = [
    Tool(
        name="getContent",
        func=get_content,
        description="Fetch and return all report page contents as a string. Input: Metadata"
    )
]

def create_knowledge_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True
    )

    return agent

if __name__ == "__main__":
    agent = create_knowledge_agent()
    metadata = {'patient_id': 'pt-1', 'report_date': '2025-09-17', 'report_id': 'RPT-4', 'confidence': 0.8543846785146648}
    prompt = f"""
    You are an information extraction agent. You will perform the following tasks:
    1. Pass the entire metadata to the getContent tool and get the complete report text.
    2. Analyse the report text and look for important information
    3. return the output in strictly the following way:
    {{findings: list of only the important details from the text about the patient's condition
    values: key value pairs of critical data with their mentioned values}}
    
    metadata: {metadata}
        """

    response = agent.invoke({"input": prompt})
    print("\n=== Extraction RESULT ===")
    print(response)
