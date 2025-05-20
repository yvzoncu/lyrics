from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from psycopg2.extras import DictCursor
from lyrics_fetcher import LyricsFetcher
from evaluator import evaluator
from dotenv import load_dotenv
from compare import text_to_emotion


load_dotenv()

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=2)
MISTRAL_API_KEY = "RXuqVFz52CqZ61kRjLWtzcMgfdoCNV3z"

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
    allow_credentials=False,  # Set to False
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=DictCursor,
    )


@app.get("/api/new-song-suggester")
async def search(query: str = None, k: int = 5):
    def search_operation():
        if not query or query.strip() == "":
            return {
                "error": "Empty query",
                "message": "Please provide a non-empty search query containing text to analyze for emotions.",
                "suggestion": "Try searching with a phrase or sentence that expresses an emotion.",
            }

        # Step 1: Evaluate if the query mentions specific songs or artists
        evaluation_result = evaluator(query)
        print(evaluation_result)

        if not evaluation_result:
            return {
                "error": "problem fetching",
            }

        if evaluation_result.get("query_type") == "custom":
            return search_by_custom_input(query, evaluation_result, k)

        return search_all_songs_by_emotion(query, k)

    return await asyncio.get_event_loop().run_in_executor(executor, search_operation)


def prompt_emotion_vector(values):
    threshold = 0.01
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    return [{label: value} for label, value in zip(labels, values) if value > threshold]


def search_all_songs_by_emotion(query, k):
    prompt_vector = text_to_emotion(query)

    result = []

    sql_query = """
        SELECT 
            s.id, 
            s.song, 
            s.artist, 
            s.full_lyric,
            s.dominants,
            s.tags,
            s.genre,
            COUNT(*) AS hits
        FROM public.lyrics lr
        JOIN public.songs s ON lr.song_id = s.id
        WHERE (lr.emotion_vector <=> %s::vector) < 0.001
        GROUP BY s.id
        ORDER BY hits DESC
        LIMIT %s;
    """

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)

    try:
        cursor.execute(sql_query, (prompt_vector, k))
        songs = cursor.fetchall()

        for song in songs:
            result.append(
                {
                    "song": song["song"],
                    "artist": song["artist"],
                    "full_lyric": song["full_lyric"],
                    "dominants": song["dominants"],
                    "tags": song["tags"],
                    "genre": song["genre"],
                    "count": song["hits"],
                }
            )

        return {"emotions": prompt_emotion_vector(prompt_vector), "items": result}

    finally:
        cursor.close()
        conn.close()


def search_by_custom_input(query, evaluation_result, k):
    prompt_vector = text_to_emotion(query)
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    max_idx = max(range(len(prompt_vector)), key=lambda i: prompt_vector[i])

    dominant_emotion_in_prompt = {
        "emotion": labels[max_idx],
        "value": round(prompt_vector[max_idx], 6),
    }

    result = []

    # Convert comma-separated strings to lists and clean whitespace
    song_list = [
        s.strip() for s in evaluation_result["song_names"].split(",") if s.strip()
    ]
    artist_list = [
        a.strip() for a in evaluation_result["artist_names"].split(",") if a.strip()
    ]
    search_list = [
        term.strip() for term in evaluation_result["search"].split(",") if term.strip()
    ]

    # Start SQL
    sql = """
        SELECT 
            s.id,
            s.song,
            s.artist,
            s.full_lyric,
            s.dominants,
            s.tags,
            s.genre
        FROM public.songs s
        
        WHERE 1=1
    """

    # Parameters list for binding
    params = []

    if song_list or artist_list or search_list:
        sql += " AND ("
        conditions = []

        if song_list:
            for song_name in song_list:
                conditions.append("s.song ILIKE %s")
                params.append(f"%{song_name}%")

        if artist_list:
            placeholders = ", ".join(["%s"] * len(artist_list))
            conditions.append(f"s.artist ILIKE ANY (ARRAY[{placeholders}])")
            params.extend([f"%{a}%" for a in artist_list])

        if search_list:
            for term in search_list:
                conditions.append("(s.tags::text ILIKE %s OR s.genre::text ILIKE %s)")
                params.extend([f"%{term}%", f"%{term}%"])

        sql += " OR ".join(conditions)
        sql += ")"

        sql += " LIMIT %s"
        params.append(k)

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)

    try:
        cursor.execute(sql, params)
        songs = cursor.fetchall()

        # If no results found with direct search, try searching by dominant emotion
        if not songs and dominant_emotion_in_prompt:
            print(
                f"No results in initial query. Falling back to emotion search for: {dominant_emotion_in_prompt['emotion']}"
            )

            # Build emotion search query
            emotion_sql = """
                SELECT
                      *,
                    (elem->> %s)::float AS emotion_value,
                    1 - abs((elem->> %s)::float - %s) AS similarity_score
                    FROM songs,
                    LATERAL jsonb_array_elements(dominants) AS elem
                    WHERE elem ? %s
                    ORDER BY similarity_score DESC
                    LIMIT %s;
            """

            cursor.execute(
                emotion_sql,
                [
                    dominant_emotion_in_prompt["emotion"],
                    dominant_emotion_in_prompt["emotion"],
                    dominant_emotion_in_prompt["value"],
                    dominant_emotion_in_prompt["emotion"],
                    k,
                ],
            )
            songs = cursor.fetchall()

        for song in songs:
            result.append(
                {
                    "song": song["song"],
                    "artist": song["artist"],
                    "full_lyric": song["full_lyric"],
                    "dominants": song["dominants"],
                    "tags": song["tags"],
                    "genre": song["genre"],
                }
            )

        return {"emotions": prompt_emotion_vector(prompt_vector), "items": result}

    finally:
        cursor.close()
        conn.close()
