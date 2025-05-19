import requests
import json
import re

MISTRAL_API_KEY = "RXuqVFz52CqZ61kRjLWtzcMgfdoCNV3z"  # Your Mistral API key


def send_mistral_request(
    user_prompt,
):
    prompt = f"""
        You are an AI that analyzes and valiates user prompts.
        You have to analyze the user prompt and respond only as defined:
        {user_prompt}

        You have 2 tasks.

        Task 1: 
        
        - If the prompt contains a specific artist or a specific song name or a specific search term,  return following response :{{"success": "S", "query_type": "custom", "song_names": "song title1, song title2", "artist_names": "artist name1, artist name2", "search":"item1, item2"}}

        
        Task 2: If prompt is not falling under above criteria;
        if the input is NOT meaningfull or violating common ethical and moral rules, return following response:{{"success": "F", "message": "Try a ifferent prompt"}}

         If the input is enough for detecting emotion using model, return following response:{{"success": "T", "query_type": "random"}}
        
        Now respond for: "{user_prompt}"
        and ONLY return one of the defined responses.
""".strip()

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": False,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Handle the response
    if response.status_code == 200:
        return response.json()
    else:
        return None


def extract_and_clean_json(content: str) -> dict:
    # Remove Markdown-style code fences like ```json or ```
    cleaned = re.sub(r"```[\w]*", "", content).strip()

    # Extract the first JSON-like object from the cleaned string
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the string.")

    json_str = match.group(0)

    # Try parsing the JSON string
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    return data


def evaluator(input):
    res = send_mistral_request(input)
    if res:
        content = res["choices"][0]["message"]["content"]
        cleaned_res = extract_and_clean_json(content)
        if cleaned_res:
            return cleaned_res
        else:
            return None
    else:
        return None


def tag_finder(lyric, song, artist):
    prompt = f"""
        You are an AI that analyzes song lyrics with years of experiene.
        Lyric: {lyric},
        Song:{song},
        Artist:{artist},
        Analyse the song lyric of song {song} by {artist} and find max 5 unique keywords.
        Use lyrics and your own knowladgebase and expertise to find best keywoords
        Keywords you suggest should be highly related to the song and should remind the song.
        Keywords should be one word.
        Return only keywords nothing else as comma seperated. 
        Fex: "Keyword1, Keyword2, Keyword3,"

""".strip()

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": False,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Handle the response
    if response.status_code == 200:
        content_json = response.json()
        content_data = content_json["choices"][0]["message"]["content"]
        word_list = [word.strip() for word in content_data.split(",")]
        return json.dumps(word_list)

    else:
        return json.dumps([])
