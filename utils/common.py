import json
import hashlib
from datetime import datetime
import logging

def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logging.error("Failed to parse JSON:", exc_info=True)
        return {}

def generate_memory_id(content: str, speaker: str = "nova") -> str:
    timestamp = datetime.now().isoformat()
    hash_input = f"{timestamp}{content}{speaker}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def model_supports_json(model_name: str) -> bool:
    return "gpt-4o" in model_name.lower()  # You can update this as OpenAI changes support
