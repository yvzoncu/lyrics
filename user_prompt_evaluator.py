import requests
import re
import json
import os


MISTRAL_API_KEY = os.getenv("MISTRAL")
MODEL_URL = 'https://api.mistral.ai/v1/chat/completions'

def build_clean_json_prompt(input: str) -> str:
    return f"""
You are an assistant that always responds with strict JSON only — no explanations or extra text.

Analyse the input and decide if provided input is enough for detecting emotion from the input using  "text-classification", model="SamLowe/roberta-base-go_emotions". 
if the input is NOT enough for detecting emotion using model, return 3 example senteces in json format.

✅ Expected JSON output if the input is NOT enough for detecting emotion from input:
{{
  "samples": [
    "I am feeling nervous today.",
    "There’s a sense of nervousness I can't shake.",
    "Im just really on edge right now."
  ]
}}

If text is enough for detecting emotion return below json. 
✅ Expected JSON output if the input is enough for detecting emotion from input::
{{
  "samples": []
}}

Now respond for: "{input}"
ONLY return the JSON object.
""".strip()


def query_mistral(input: str):
    prompt = build_clean_json_prompt(input=input)
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistral-medium",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }
    
    response = requests.post(MODEL_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
    

def extract_json(raw: str):
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("No valid JSON found in Mistral response")
    
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    
def clean_samples_json(data: dict) -> dict:
    # Only allow: letters, numbers, space, . , ! ? ' -
    allowed_pattern = r"[^a-zA-Z0-9 .,!?'\-]"
    
    cleaned_samples = [
        re.sub(allowed_pattern, "", sample)
        for sample in data.get("samples", [])
    ]
    return {"samples": cleaned_samples}

def evaluator(input: str) -> list[str]:
    res = query_mistral(input=input)
    raw = res["choices"][0]["message"]["content"]
    jsonMatch = extract_json(raw)
    cleaned_json = clean_samples_json(jsonMatch)
    return cleaned_json['samples']


