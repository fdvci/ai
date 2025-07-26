# utils/json_helper.py
"""Helper module to handle JSON responses without response_format parameter"""

import json
import re
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text response, handling various formats"""
    try:
        # First try direct JSON parsing
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON in code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Try to find raw JSON object
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except:
            continue
    
    # Try to extract JSON from the entire text after cleaning
    cleaned = text.strip()
    if cleaned.startswith('{') and cleaned.endswith('}'):
        try:
            return json.loads(cleaned)
        except:
            pass
    
    logger.warning(f"Could not extract JSON from text: {text[:200]}...")
    return None

def safe_json_request(client, messages: list, default_response: Dict[str, Any], 
                     model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
    """Make a request expecting JSON response, with fallback handling"""
    try:
        # First ensure the system message asks for JSON
        if messages and messages[0]["role"] == "system":
            if "JSON" not in messages[0]["content"]:
                messages[0]["content"] += "\nAlways respond with valid JSON only, no other text."
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        content = response.choices[0].message.content.strip()
        result = extract_json_from_text(content)
        
        if result:
            return result
        else:
            logger.warning("Failed to parse JSON, using default response")
            return default_response
            
    except Exception as e:
        logger.error(f"Error in JSON request: {e}")
        return default_response