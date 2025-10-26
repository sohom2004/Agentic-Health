from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import convert_to_jpg, get_ocr, get_file_type
import os
from dotenv import load_dotenv

load_dotenv()

tools = [
    Tool(
        name="FileTypeDetector",
        func=get_file_type,
        description="Detects if a given file path is a PDF or an image. Input: file path (string)."
    ),
    Tool(
        name="PDFtoImage",
        func=convert_to_jpg,
        description="Converts PDF into a list of image paths. Input: file path (string)."
    ),
    Tool(
        name="OCRTool",
        func=get_ocr,
        description="Runs OCR on a list of image paths. Input: list of image paths."
    )
]

def create_ocr_agent():
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
    agent = create_ocr_agent()
    file_path = "./sample-report.pdf"
    prompt = f"""
    You are an OCR extraction assistant.
    Steps you MUST follow:
    1. Detect file type using FileTypeDetector.
    2. If PDF, convert to images using PDFtoImages.
    3. Run OCRTool on the image paths.
    4. Return the content and confidence from the OCRTool.
    File: {file_path}
        """

    response = agent.invoke({"input": prompt})
    print("\n=== OCR RESULT ===")
    print(response)