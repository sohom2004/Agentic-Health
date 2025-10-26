from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import transcribe
import os
from dotenv import load_dotenv

load_dotenv()

tools = [
    Tool(
        name="SpeechToText",
        func=transcribe,
        description="Transcribes audio into text. Input: file path (string)"
    )
]

def create_stt_agent():
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
    agent = create_stt_agent()
    file_path = "./Sample.mp3"
    prompt = f"""
    You are a text extraction agent. You will perform text transcription on the audio file and return the text content.
    File: {file_path}
        """

    response = agent.invoke({"input": prompt})
    print("\n=== STT RESULT ===")
    print(response)
