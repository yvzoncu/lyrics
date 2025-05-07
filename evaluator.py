import requests
import json

MISTRAL_API_KEY = 'RXuqVFz52CqZ61kRjLWtzcMgfdoCNV3z'  # Your Mistral API key

def send_mistral_request(user_prompt):
    prompt = f"""
        You are an AI that analyzes user prompts about songs and music.
        Analyze the user prompt and respond only as JSON:
        {user_prompt}
        - If the prompt mentions a specific song, the response must be in the following format:
          {{"query_type": "song", "song_names": "song title1, song title2", "artist_names": ""}}
        - If the prompt mentions a specific artist, the response must be in the following format:
          {{"query_type": "artist", "song_names": "", "artist_names": "artist name1, artist name2"}}
        - If the prompt mentions both an artist and a song, the response must be in the following format:
          {{"query_type": "artist_and_song", "song_names": "song title1, song title2", "artist_names": "artist name1, artist name2"}}  
        - If the prompt is unclear or doesn't mention a song or artist, response must be in the following format: {{"query_type": "random"}}

        The song or artist name should be extracted and returned in lowercase if possible, and be as exact as possible to what was mentioned in the user's prompt. Ignore extra words and focus on identifying the song title or artist name in the query.
    """
    
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
        json_str = response['choices'][0]['message']['content']
        
        # Clean by removing newlines and any extra whitespace
        json_str = json_str.replace('\n', '').replace('\\n', '')
        
        # Parse the JSON string to a dictionary
        json_data = json.loads(json_str)
        
        return json_data
    except (KeyError, json.JSONDecodeError) as e:
        return None 
    

def evaluator(input):
    res = send_mistral_request(input)
    if res:
        res = extract_clean_json(res)
        if res:
            return res
        else:
            return None
    else:
        return None

    
    
