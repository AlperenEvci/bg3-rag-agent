from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# For langchain's ChatOpenAI, we need to set the GROQ_API_KEY env variable
# This is because older versions of LangChain use this environment variable
os.environ["GROQ_API_KEY"] = groq_api_key

# Create LLM with explicit configuration for Groq API
llm = ChatOpenAI(
    openai_api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
)