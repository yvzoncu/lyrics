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

load_dotenv() 

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=4)

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
        # Handle empty query case
        if not query or query.strip() == "":
            return {
                "error": "Empty query",
                "message": "Please provide a non-empty search query containing text to analyze for emotions.",
                "suggestion": "Try searching with a phrase or sentence that expresses an emotion."
            }     
        
        try:
            # Try to generate the emotion vector and catch any potential errors
            query_vec = get_emotion_vector(query).reshape(1, -1).astype('float32')
            query_emotions = emotion_model(query)[0]
            sorted_query_emotions = sorted(query_emotions, key=lambda x: x['score'], reverse=True)
            
            # Check if the vector was generated successfully
            if np.isnan(query_vec).any() or np.isinf(query_vec).any() or np.linalg.norm(query_vec) < 1e-10:
                return {
                    "error": "Invalid emotion vector",
                    "message": "Could not generate a valid emotion vector from your query.",
                    "suggestion": "Try using a different query with clearer emotional content."
                }
                
            # Early return if no lyrics in the database
            if not lyric_ids:
                return {
                    "query": query, 
                    "query_emotions": sorted_query_emotions[:5], 
                    "results": [],
                    "message": "No lyrics in the database to search against."
                }

            # Proceed with search as normal
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
                vec = pickle.loads(row['embedding'])
                all_labels = list(emotion_model.model.config.id2label.values())
                
                results.append({
                    "song": row['song'],
                    "lyric": row['lyric'],
                    "similarity_ratio": float(distances[0][i])
           
                })

            cursor.close()
            conn.close()
            
            # Add a message if no results were found
            if not results:
                return {
                    "query": query, 
                    "query_emotions": sorted_query_emotions[:5], 
                    "results": [],
                    "message": "No matching lyrics found for the emotional content of your query."
                }
            
            song_scores = defaultdict(list)

            for entry in results:
                song_scores[entry["song"]].append(entry["similarity_ratio"])
            
            aggregated = []

            for song, scores in song_scores.items():
                sorted_scores = sorted(scores, reverse=True)
                top_scores = sorted_scores[:100]
                average_top = sum(top_scores) / len(top_scores)
                max_score = max(scores)
                total = sum(top_scores)
                score = round((average_top + max_score + total) / 3, 5) 

                aggregated.append({
                    "song": song,
                    "average_top_similarity": round(average_top, 5),
                    "max_similarity": round(max_score, 5),
                    "total_top_similarity": round(total, 5),
                    "final_score": score,
                    "count": len(scores)
                                })

                # Step 3: Rank songs by final_score
            ranked = sorted(aggregated, key=lambda x: x["final_score"], reverse=True)
                
            return {"query": query, "query_emotions": sorted_query_emotions[:5], "results": ranked}
            
        except Exception as e:
            # Catch any unexpected errors during vector generation or search
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
            SELECT song, artist FROM songs
            WHERE LOWER(song) LIKE %s
        """, (query_song,))

        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    rows = await asyncio.get_event_loop().run_in_executor(executor, db_operation)
    return {"songs": [{"song": row[0], "artist": row[1]} for row in rows]}


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