from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text):
    res = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return res.data[0].embedding
