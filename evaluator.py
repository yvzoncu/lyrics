import requests
import json
import re

MISTRAL_API_KEY = 'RXuqVFz52CqZ61kRjLWtzcMgfdoCNV3z'  # Your Mistral API key

def send_mistral_request(user_prompt):
    prompt = f"""
        You are an AI that analyzes and valiates user prompts.
        You have to analyze the user prompt and respond only as defined:
        {user_prompt}

        You have 2 tasks.

        Task 1: 
        - If the prompt contains of only a specific song, the response must be in the following format: fex:"Billie Eilish"
          {{"success": "S", "query_type": "song", "song_names": "song title1, song title2", "artist_names": ""}}
         
        - If the prompt contains of only a specific artist, the response must be in the following format: fex:"Happier than ever"
          {{"success": "S", "query_type": "artist", "song_names": "", "artist_names": "artist name1, artist name2"}}

        - If the prompt contains of only a specific artist and a song, the response must be in the following format:fex:"Billie Eilish Happier than ever"
          {{"success": "S", "query_type": "artist_and_song", "song_names": "song title1, song title2", "artist_names": "artist name1, artist name2"}}  

        - If the prompt contains a specific artist and other additional explanatoy text, the response must be in the following format: fex:"Im sad and want to listen Billie Eilish or Ariana Grande"
          {{"success": "S", "query_type": "artist_emotion", "song_names": "", "artist_names": "artist name1, artist name2"}}

        
        Task 2: If prompt is not falling under above criteria;
        if the input is NOT meaningfull or violating common ethical and moral rules, the response must be in the following format.
        {{"success": "F", "message": "Try a ifferent prompt"}}

         If the input is enough for detecting emotion using model, the response must be in the following format. 
         {{"success": "T", "query_type": "random"}}
        
        Now respond for: "{user_prompt}"
        and ONLY return one of the defined JSON object.
""".strip()     
    
    
    url = 'https://api.mistral.ai/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {MISTRAL_API_KEY}',
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'mistral-medium',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.7,
        'stream': False,
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Handle the response
    if response.status_code == 200:
        return response.json()
    else:
        return None



def extract_clean_json(response):
    try:
        content = response['choices'][0]['message']['content']

        # Look for the first curly-brace-enclosed JSON object at the start
        json_match = re.search(r'^\s*({.*?})\s*(?:\n|$)', content, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)

            # Clean escaped characters (optional if you trust model output)
            json_str = json_str.replace('\\_', '_')

            # Try loading the JSON
            parsed = json.loads(json_str)

            if "success" in parsed:
                return parsed

        return None
    except (KeyError, json.JSONDecodeError, TypeError):
        return None




def evaluator(input):
    res = send_mistral_request(input)
    print(res)
    if res:
        cleaned_res = extract_clean_json(res)
        if cleaned_res:
            return cleaned_res
        else:
            return None
    else:
        return None

    
    
