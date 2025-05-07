from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import psycopg2
import numpy as np
import pickle
import faiss
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import os
from psycopg2.extras import DictCursor
from lyrics_fetcher import LyricsFetcher
from collections import defaultdict
from user_prompt_evaluator import evaluator
from dotenv import load_dotenv
import requests
import json

load_dotenv() 

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=2)
MISTRAL_API_KEY = 'RXuqVFz52CqZ61kRjLWtzcMgfdoCNV3z'  

lyrics_fetcher = LyricsFetcher() 

# Environment variables
DB_HOST = os.getenv("DB_HOST", "moodify.cje0wa8qioij.eu-north-1.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "lyricsdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SongCreate(BaseModel):
    song: str
    artist: str
    full_lyric: str = None

class TextInput(BaseModel):
    text: str

class LyricLine(BaseModel):
    lyric: str

class UploadLyricsRequest(BaseModel):
    song_id: int
    lyrics: List[LyricLine]

class SongRequest(BaseModel):
    song: str
    artist: str
    contributor:int


emotion_model = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

def get_emotion_vector(text):
    results = emotion_model(text)[0]
    label_scores = {res['label']: res['score'] for res in results}
    all_labels = list(emotion_model.model.config.id2label.values())
    vec = np.array([label_scores.get(label, 0.0) for label in all_labels], dtype=np.float32)
    return vec / np.linalg.norm(vec)

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=DictCursor
    )

conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("SELECT id, embedding FROM lyrics WHERE embedding IS NOT NULL")
rows = cursor.fetchall()
cursor.close()
conn.close()

lyric_ids = []
vectors = []
for row in rows:
    lyric_ids.append(row['id'])
    vectors.append(pickle.loads(row['embedding']))

if vectors:
    vectors = np.vstack(vectors).astype('float32')
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
else:
    dimension = 28
    index = faiss.IndexFlatIP(dimension)
    lyric_ids = []

@app.get("/api/all-songs")
def list_songs(skip: int = 0, limit: int = 100):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT song, artist 
        FROM songs 
        ORDER BY song, artist 
        LIMIT %s OFFSET %s""", (limit, skip))

        songs = cursor.fetchall()
        return {"tottal": len(songs), "songs": songs}
    finally:
        cursor.close()
        conn.close()



@app.get("/api/lyrics")
async def list_all_lyrics(skip: int = 0, limit: int = 100):
    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id AS song_id, s.song, s.artist, l.id AS lyric_id, l.lyric
            FROM songs s
            JOIN lyrics l ON l.song_id = s.id
            ORDER BY l.id
            LIMIT %s OFFSET %s
        """, (limit, skip))
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        cursor.execute("SELECT COUNT(*) FROM lyrics")
        total = cursor.fetchone()['count']
        cursor.close()
        conn.close()
        return {"total": total, "skip": skip, "limit": limit, "lyrics": results}
    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/song-suggester")
async def search(query: str = None, k: int = 5):
    def search_operation():
        if not query or query.strip() == "":
            return {
                "error": "Empty query",
                "message": "Please provide a non-empty search query containing text to analyze for emotions.",
                "suggestion": "Try searching with a phrase or sentence that expresses an emotion."
            }
        


        try:
            query_vec = get_emotion_vector(query).reshape(1, -1).astype('float32')
            query_emotions = emotion_model(query)[0]
            sorted_query_emotions = sorted(query_emotions, key=lambda x: x['score'], reverse=True)

            if np.isnan(query_vec).any() or np.isinf(query_vec).any() or np.linalg.norm(query_vec) < 1e-10:
                return {
                    "error": "Invalid emotion vector",
                    "message": "Could not generate a valid emotion vector from your query.",
                    "suggestion": "Try using a different query with clearer emotional content."
                }

            if not lyric_ids:
                return {
                    "query": query,
                    "query_emotions": sorted_query_emotions[:5],
                    "results": [],
                    "message": "No lyrics in the database to search against."
                }

            distances, indices = index.search(query_vec, k)
            conn = get_db_connection()
            cursor = conn.cursor()
            results = []

            for i, idx in enumerate(indices[0]):
                if idx >= len(lyric_ids):
                    continue
                lyric_id = lyric_ids[idx]
                cursor.execute("""
                    SELECT l.id AS lyric_id, l.lyric, s.song, s.artist, l.embedding
                    FROM lyrics l
                    JOIN songs s ON l.song_id = s.id
                    WHERE l.id = %s
                """, (lyric_id,))
                row = cursor.fetchone()
                if not row:
                    continue

                results.append({
                    "song": row['song'],
                    "artist": row['artist'],
                    "lyric": row['lyric'],
                    "similarity_ratio": float(distances[0][i])
                })

            cursor.close()
            conn.close()

            if not results:
                return {
                    "query": query,
                    "query_emotions": sorted_query_emotions[:5],
                    "results": [],
                    "message": "No matching lyrics found for the emotional content of your query."
                }

            # --- Updated scoring logic with artist retained ---
            song_scores = defaultdict(lambda: {"artist": None, "scores": []})

            for entry in results:
                song = entry["song"]
                artist = entry["artist"]
                score = entry["similarity_ratio"]
                song_scores[song]["scores"].append(score)
                song_scores[song]["artist"] = artist

            aggregated = []

            for song, data in song_scores.items():
                scores = data["scores"]
                artist = data["artist"]
                sorted_scores = sorted(scores, reverse=True)
                top_scores = sorted_scores[:100]
                average_top = sum(top_scores) / len(top_scores)
                max_score = max(scores)
                total = sum(top_scores)
                final_score = round((average_top + max_score + total) / 3, 5)

                aggregated.append({
                    "song": song,
                    "artist": artist,
                    "average_top_similarity": round(average_top, 5),
                    "max_similarity": round(max_score, 5),
                    "total_top_similarity": round(total, 5),
                    "final_score": final_score,
                    "count": len(scores)
                })

            ranked = sorted(aggregated, key=lambda x: x["final_score"], reverse=True)

            return {
                "query": query,
                "query_emotions": sorted_query_emotions[:3],
                "results": ranked
            }

        except Exception as e:
            return {
                "error": "Search error",
                "message": f"An error occurred while processing your search: {str(e)}",
                "suggestion": "Try simplifying your query or check if the service is working correctly."
            }

    return await asyncio.get_event_loop().run_in_executor(executor, search_operation)



@app.get("/api/find-song-by-name")
async def check_song_exists(song: str = Query(...)):
    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor()

        query_song = f"%{song.strip().lower()}%"

        cursor.execute("""
            SELECT song, artist, tags, full_lyric FROM songs
            WHERE LOWER(song) LIKE %s
        """, (query_song,))

        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    rows = await asyncio.get_event_loop().run_in_executor(executor, db_operation)
    return {"songs": [{"song": row[0], "artist": row[1], "tags": row[2], "lyrics": row[3]} for row in rows]}


@app.post("/api/predict")
def predict_emotions(input: TextInput):
    # Get validation results from the evaluator API
    validation_result = evaluator(input=input)
    
    # Check if validation result indicates an error
    if isinstance(validation_result, list) and len(validation_result) > 0:
        return {
            "success": False,
            "error": "No emotion",
            "message": validation_result,
            "suggestion": "Try using a different prompt with clearer emotional content."
        }    

    # No errors found, proceed with emotion analysis
    # Assuming emotion_model is async
    results = emotion_model(input.text)
    # Make sure we're accessing the results correctly
    emotions = results[0] if isinstance(results, list) else results
    sorted_results = sorted(emotions, key=lambda r: r["score"], reverse=True)
    top_emotions = sorted_results[:3]
    
    return {"success": True, "text": input.text, "emotions": top_emotions}


@app.post("/api/fetch-and-process")
async def fetch_and_process(request: SongRequest = Body(...)):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Step 1: Check if song already exists
        cursor.execute("""
            SELECT id, song, artist FROM songs
            WHERE song = %s AND artist = %s
        """, (request.song, request.artist))
        existing = cursor.fetchone()
        
        if existing:
            return {
                "message": "Song already exists",
                "id": existing["id"],
                "song": existing["song"],
                "artist": existing["artist"]
            }

        # Step 2: Fetch lyrics if not in DB
        data = lyrics_fetcher.get_lyrics(request.song, request.artist)

        if not data or not data.get("lyrics"):
            raise HTTPException(status_code=404, detail="Lyrics not found.")
        
        contr = lyrics_fetcher.get_contributors(data["lyrics"])

        if contr < request.contributor:
            raise HTTPException(status_code=404, detail=f"Not enough contributors. Found: {contr}")
        
        lyrics = lyrics_fetcher.extract_lyrics(data["lyrics"])
        full_lyrics = "\n".join(lyrics) 
       
        # Step 3: Insert song into database
        cursor.execute("""
            INSERT INTO songs (song, artist, full_lyric)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (request.song, request.artist, full_lyrics))
        song_id = cursor.fetchone()["id"]
        conn.commit()

        # Step 4: Process lyrics and create embeddings
        def process_lyrics():
            new_embeddings = []
            new_ids = []
            
            # Generate embeddings for each lyric line
            embeddings = [get_emotion_vector(lyric) for lyric in lyrics]
            
            conn_inner = get_db_connection()
            cursor_inner = conn_inner.cursor()
            
            for lyric, emb in zip(lyrics, embeddings):
                cursor_inner.execute("""
                    INSERT INTO lyrics (song_id, lyric, embedding)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (song_id, lyric, psycopg2.Binary(pickle.dumps(emb))))
                lyric_id = cursor_inner.fetchone()['id']
                conn_inner.commit()
                new_embeddings.append(emb)
                new_ids.append(lyric_id)
            
            if new_embeddings:
                index.add(np.array(new_embeddings, dtype='float32'))
                lyric_ids.extend(new_ids)
            
            cursor_inner.close()
            conn_inner.close()
            return len(new_ids)
        
        # Run the lyric processing in the background
        lyrics_count = await asyncio.get_event_loop().run_in_executor(executor, process_lyrics)

        return {
            "message": "Song added and processed",
            "id": song_id,
            "song": request.song,
            "artist": request.artist,
            "lyrics_count": lyrics_count,
            "full_lyrics": full_lyrics
        }

    finally:
        cursor.close()
        conn.close()



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
          - If the prompt is unclear or doesn't mention a song or artist, return: {{"query_type": "random"}}
        
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

@app.get("/api/new-song-suggester")
async def search(query: str = None, k: int = 5):
    def search_operation():
        if not query or query.strip() == "":
            return {
                "error": "Empty query",
                "message": "Please provide a non-empty search query containing text to analyze for emotions.",
                "suggestion": "Try searching with a phrase or sentence that expresses an emotion."
            }
        
        # First, evaluate if the query mentions specific songs or artists
        evaluation_result = evaluator(query)
        
        # If evaluation returns None or indicates a random query, process by emotion
        if evaluation_result is None or evaluation_result.get('query_type') == 'random':
            return search_by_emotion(query, k)
        else:
            # Process specific song/artist queries
            return search_by_specifics(query, evaluation_result, k)
    
    return await asyncio.get_event_loop().run_in_executor(executor, search_operation)

def search_by_emotion(query, k):
    try:
        query_vec = get_emotion_vector(query).reshape(1, -1).astype('float32')
        query_emotions = emotion_model(query)[0]
        sorted_query_emotions = sorted(query_emotions, key=lambda x: x['score'], reverse=True)

        if np.isnan(query_vec).any() or np.isinf(query_vec).any() or np.linalg.norm(query_vec) < 1e-10:
            return {
                "error": "Invalid emotion vector",
                "message": "Could not generate a valid emotion vector from your query.",
                "suggestion": "Try using a different query with clearer emotional content."
            }

        if not lyric_ids:
            return {
                "query": query,
                "query_emotions": sorted_query_emotions[:5],
                "results": [],
                "message": "No lyrics in the database to search against."
            }

        distances, indices = index.search(query_vec, k)
        conn = get_db_connection()
        cursor = conn.cursor()
        results = []

        for i, idx in enumerate(indices[0]):
            if idx >= len(lyric_ids):
                continue
            lyric_id = lyric_ids[idx]
            cursor.execute("""
                SELECT l.id AS lyric_id, l.lyric, s.song, s.artist, l.embedding
                FROM lyrics l
                JOIN songs s ON l.song_id = s.id
                WHERE l.id = %s
            """, (lyric_id,))
            row = cursor.fetchone()
            if not row:
                continue

            results.append({
                "song": row['song'],
                "artist": row['artist'],
                "lyric": row['lyric'],
                "similarity_ratio": float(distances[0][i])
            })

        cursor.close()
        conn.close()

        if not results:
            return {
                "query": query,
                "query_emotions": sorted_query_emotions[:5],
                "results": [],
                "message": "No matching lyrics found for the emotional content of your query."
            }

        # --- Scoring logic with artist retained ---
        song_scores = defaultdict(lambda: {"artist": None, "scores": []})

        for entry in results:
            song = entry["song"]
            artist = entry["artist"]
            score = entry["similarity_ratio"]
            song_scores[song]["scores"].append(score)
            song_scores[song]["artist"] = artist

        aggregated = []

        for song, data in song_scores.items():
            scores = data["scores"]
            artist = data["artist"]
            sorted_scores = sorted(scores, reverse=True)
            top_scores = sorted_scores[:100]
            average_top = sum(top_scores) / len(top_scores)
            max_score = max(scores)
            total = sum(top_scores)
            final_score = round((average_top + max_score + total) / 3, 5)

            aggregated.append({
                "song": song,
                "artist": artist,
                "average_top_similarity": round(average_top, 5),
                "max_similarity": round(max_score, 5),
                "total_top_similarity": round(total, 5),
                "final_score": final_score,
                "count": len(scores)
            })

        ranked = sorted(aggregated, key=lambda x: x["final_score"], reverse=True)

        return {
            "query": query,
            "query_emotions": sorted_query_emotions[:3],
             "extracted_info": None,
            "results": ranked,
            "search_type": "emotion"
        }

    except Exception as e:
        return {
            "error": "Search error",
            "message": f"An error occurred while processing your emotion search: {str(e)}",
            "suggestion": "Try simplifying your query or check if the service is working correctly."
        }

def search_by_specifics(query, evaluation_result, k=10):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query_type = evaluation_result.get('query_type')
        results = []

        if query_type == 'song' or query_type == 'artist_and_song':
            song_names = evaluation_result.get('song_names', '').split(', ')
            song_conditions = []
            song_params = []
            
            for song_name in song_names:
                if song_name.strip():
                    song_conditions.append("LOWER(s.song) LIKE %s")
                    song_params.append(f"%{song_name.lower().strip()}%")
            
            if song_conditions:
                song_query = f"""
                    SELECT s.song, s.artist
                    FROM songs s
                    WHERE {' OR '.join(song_conditions)}
                    LIMIT {k}
                """
                cursor.execute(song_query, song_params)
                song_results = cursor.fetchall()
                
                for row in song_results:
                    results.append({
                        "song": row['song'],
                        "artist": row['artist'],
                        "match_type": "song"
                    })
        
        if query_type == 'artist' or query_type == 'artist_and_song':
            artist_names = evaluation_result.get('artist_names', '').split(', ')
            artist_conditions = []
            artist_params = []
            
            for artist_name in artist_names:
                if artist_name.strip():
                    artist_conditions.append("LOWER(s.artist) LIKE %s")
                    artist_params.append(f"%{artist_name.lower().strip()}%")
            
            if artist_conditions:
                artist_query = f"""
                    SELECT s.song, s.artist
                    FROM songs s
                    WHERE {' OR '.join(artist_conditions)}
                    GROUP BY s.song, s.artist
                    LIMIT {k}
                """
                cursor.execute(artist_query, artist_params)
                artist_results = cursor.fetchall()
                
                for row in artist_results:
                    results.append({
                        "song": row['song'],
                        "artist": row['artist'],
                        "match_type": "artist"
                    })
        
        cursor.close()
        conn.close()

        if not results:
            # Fallback to emotion-based search if no direct matches
            return search_by_emotion(query, k)
        
        query_emotions = emotion_model(query)[0]
        sorted_query_emotions = sorted(query_emotions, key=lambda x: x['score'], reverse=True)

        return {

            "query": query,
            "query_emotions": sorted_query_emotions[:3],
            "query_type": query_type,
             "extracted_info": {
                "songs": evaluation_result.get('song_names', ''),
                "artists": evaluation_result.get('artist_names', '')
            },
            "results": sorted(results, key=lambda x: x["song"]),
            "search_type": "specific"
        }

    except Exception as e:
        return {
            "error": "Search error",
            "message": f"An error occurred while processing your specific search: {str(e)}",
            "suggestion": "Try using a more general query or check if the service is working correctly."
        }
    



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
          - If the prompt is unclear or doesn't mention a song or artist, return: {{"query_type": "random"}}
        
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

@app.get("/new-new-api/song-suggester")
async def search(query: str = None, k: int = 5):
    def search_operation():
        if not query or query.strip() == "":
            return {
                "error": "Empty query",
                "message": "Please provide a non-empty search query containing text to analyze for emotions.",
                "suggestion": "Try searching with a phrase or sentence that expresses an emotion."
            }
        
        # Step 1: Evaluate if the query mentions specific songs or artists
        evaluation_result = evaluator(query)
        
        # Step 2: Calculate the emotion vector for the query (for all cases)
        try:
            query_vec = get_emotion_vector(query).reshape(1, -1).astype('float32')
            query_emotions = emotion_model(query)[0]
            sorted_query_emotions = sorted(query_emotions, key=lambda x: x['score'], reverse=True)
            
            if np.isnan(query_vec).any() or np.isinf(query_vec).any() or np.linalg.norm(query_vec) < 1e-10:
                return {
                    "error": "Invalid emotion vector",
                    "message": "Could not generate a valid emotion vector from your query.",
                    "suggestion": "Try using a different query with clearer emotional content."
                }
                
            # Step 3: Process based on evaluation result
            if evaluation_result is None or evaluation_result.get('query_type') == 'random':
                # For random queries, find related songs by emotion from all songs
                return search_all_songs_by_emotion(query, query_vec, sorted_query_emotions, k)
            else:
                # For specific queries, get songs from DB and rank by emotion similarity
                return search_specific_songs_by_emotion(query, evaluation_result, query_vec, sorted_query_emotions, k)
                
        except Exception as e:
            return {
                "error": "Emotion processing error",
                "message": f"An error occurred while processing emotions: {str(e)}",
                "suggestion": "Try rephrasing your query."
            }
    
    return await asyncio.get_event_loop().run_in_executor(executor, search_operation)



def search_all_songs_by_emotion(query, query_vec, sorted_query_emotions, k):
    """Find related songs by emotion from all songs in the database"""
    try:
        if not lyric_ids:
            return {
                "query": query,
                "query_emotions": sorted_query_emotions[:5],
                "results": [],
                "message": "No lyrics in the database to search against."
            }

        distances, indices = index.search(query_vec, k)
        conn = get_db_connection()
        cursor = conn.cursor()
        results = []

        for i, idx in enumerate(indices[0]):
            if idx >= len(lyric_ids):
                continue
            lyric_id = lyric_ids[idx]
            cursor.execute("""
                SELECT l.id AS lyric_id, l.lyric, s.song, s.artist, l.embedding
                FROM lyrics l
                JOIN songs s ON l.song_id = s.id
                WHERE l.id = %s
            """, (lyric_id,))
            row = cursor.fetchone()
            if not row:
                continue

            results.append({
                "song": row['song'],
                "artist": row['artist'],
                "lyric": row['lyric'],
                "similarity_ratio": float(distances[0][i])
            })

        cursor.close()
        conn.close()

        if not results:
            return {
                "query": query,
                "query_emotions": sorted_query_emotions[:5],
                "results": [],
                "message": "No matching lyrics found for the emotional content of your query."
            }

        # --- Scoring logic with artist retained ---
        song_scores = defaultdict(lambda: {"artist": None, "scores": []})

        for entry in results:
            song = entry["song"]
            artist = entry["artist"]
            score = entry["similarity_ratio"]
            song_scores[song]["scores"].append(score)
            song_scores[song]["artist"] = artist

        aggregated = []

        for song, data in song_scores.items():
            scores = data["scores"]
            artist = data["artist"]
            sorted_scores = sorted(scores, reverse=True)
            top_scores = sorted_scores[:100]
            average_top = sum(top_scores) / len(top_scores)
            max_score = max(scores)
            total = sum(top_scores)
            final_score = round((average_top + max_score + total) / 3, 5)

            aggregated.append({
                "song": song,
                "artist": artist,
                "average_top_similarity": round(average_top, 5),
                "max_similarity": round(max_score, 5),
                "total_top_similarity": round(total, 5),
                "final_score": final_score,
                "count": len(scores)
            })

        ranked = sorted(aggregated, key=lambda x: x["final_score"], reverse=True)

        return {
            "query": query,
            "query_emotions": sorted_query_emotions[:3],
            "results": ranked,
            "search_type": "emotion_all_songs"
        }

    except Exception as e:
        return {
            "error": "Search error",
            "message": f"An error occurred while processing your emotion search: {str(e)}",
            "suggestion": "Try simplifying your query or check if the service is working correctly."


        }
    
def parse_embedding(raw_embedding):
    if isinstance(raw_embedding, memoryview):
        raw_embedding = raw_embedding.tobytes()
    return pickle.loads(raw_embedding)



def search_specific_songs_by_emotion(query, evaluation_result, query_vec, sorted_query_emotions, k):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query_type = evaluation_result.get('query_type')
        song_ids = set()

        if query_type in ['song', 'artist_and_song']:
            song_names = evaluation_result.get('song_names', '').split(', ')
            for name in song_names:
                cursor.execute("SELECT id FROM songs WHERE LOWER(song) LIKE %s", (f"%{name.strip().lower()}%",))
                song_ids.update(row['id'] for row in cursor.fetchall())

        if query_type in ['artist', 'artist_and_song']:
            artist_names = evaluation_result.get('artist_names', '').split(', ')
            for name in artist_names:
                cursor.execute("SELECT id FROM songs WHERE LOWER(artist) LIKE %s", (f"%{name.strip().lower()}%",))
                song_ids.update(row['id'] for row in cursor.fetchall())

        if not song_ids:
            search_all_songs_by_emotion(query, query_vec, sorted_query_emotions, k)

        cursor.execute("""
            SELECT l.id, l.embedding
            FROM lyrics l
            JOIN songs s ON l.song_id = s.id
            WHERE l.song_id = ANY(%s)
        """, (list(song_ids),))
        lyrics = cursor.fetchall()

        print(song_ids)

        if not lyrics:
            return {
                "query": query,
                "query_emotions": sorted_query_emotions[:3],
                "results": [],
                "message": "No lyrics found for the specified songs/artists."
            }

        embeddings = np.array([parse_embedding(row['embedding'])for row in lyrics]).astype('float32')
        local_index = faiss.IndexFlatL2(embeddings.shape[1])
        local_index.add(embeddings)
        distances, indices = local_index.search(query_vec, k)

        conn = get_db_connection()
        cursor = conn.cursor()
        results = []
        ly_ids = []
        for row in lyrics:
            ly_ids.append(row['id'])

        for i, idx in enumerate(indices[0]):
            if idx >= len(ly_ids):
                continue
            ly_id = ly_ids[idx]
            cursor.execute("""
                SELECT l.id AS lyric_id, l.lyric, s.song, s.artist, l.embedding
                FROM lyrics l
                JOIN songs s ON l.song_id = s.id
                WHERE l.id = %s
            """, (ly_id,))
            row = cursor.fetchone()
            if not row:
                continue

            results.append({
                "song": row['song'],
                "artist": row['artist'],
                "lyric": row['lyric'],
                "similarity_ratio": float(distances[0][i])
            })

        cursor.close()
        conn.close()

        song_scores = defaultdict(lambda: {"artist": None, "scores": []})

        for entry in results:
            song = entry["song"]
            artist = entry["artist"]
            score = entry["similarity_ratio"]
            song_scores[song]["scores"].append(score)
            song_scores[song]["artist"] = artist

        aggregated = []

        for song, data in song_scores.items():
            scores = data["scores"]
            artist = data["artist"]
            sorted_scores = sorted(scores, reverse=True)
            top_scores = sorted_scores[:100]
            average_top = sum(top_scores) / len(top_scores)
            max_score = max(scores)
            total = sum(top_scores)
            final_score = round((average_top + max_score + total) / 3, 5)

            aggregated.append({
                "song": song,
                "artist": artist,
                "average_top_similarity": round(average_top, 5),
                "max_similarity": round(max_score, 5),
                "total_top_similarity": round(total, 5),
                "final_score": final_score,
                "count": len(scores)
            })

        ranked = sorted(aggregated, key=lambda x: x["final_score"], reverse=True)

        return {
            "query": query,
            "query_emotions": sorted_query_emotions[:3],
            "extracted_info": {
                "songs": evaluation_result.get('song_names', ''),
                "artists": evaluation_result.get('artist_names', '')
            },
            "results": ranked,
            "search_type": "filtered_emotion"
        }

    except Exception as e:
        return {
            "error": "Filtered search error",
            "message": str(e),
            "suggestion": "Try using a more specific query or retry later."
        }




