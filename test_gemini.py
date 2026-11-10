
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

model_name = os.getenv("GEMINI_MODEL")

print("Testing model:", model_name)

response = client.models.generate_content(
    model=model_name,
    contents="Explain RAG in three simple sentences."
)

print("\nGemini response:")
print(response.text)


