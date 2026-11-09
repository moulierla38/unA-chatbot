
import os
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Ask Gemini a question
response = client.interactions.create(
    model="gemini-3.8-flash",
    input="Introduce yourself as the AI assistant of unA."
)

print(response.output_text)
