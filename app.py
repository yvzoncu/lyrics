from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import json
from psycopg2.extras import DictCursor, Json
from lyrics_fetcher import LyricsFetcher
from evaluator import evaluator
from dotenv import load_dotenv
from compare import text_to_emotion
from pydantic import BaseModel
from typing import List, Dict, Any


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
async def search(query: str, user_id: str, k: int = 5):
    def search_operation():

        if not query or query.strip() == "":
            return {
                "error": "Empty query",
                "message": "Please provide a non-empty search query containing text to analyze for emotions.",
                "suggestion": "Try searching with a phrase or sentence that expresses an emotion.",
            }

        val = insert_user_query(user_id, query)

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


def insert_user_query(user_id, query):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)

    sql_query = """
        INSERT INTO user_queries (user_id, query)
        VALUES (%s, %s)
        RETURNING id;
    """
    try:
        cursor.execute(sql_query, (user_id, query))
        query_id = cursor.fetchone()["id"]
        conn.commit()
        return query_id
    finally:
        cursor.close()
        conn.close()


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
                    "song_id": song["id"],
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
                    "song_id": song["id"],
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


class PlaylistItem(BaseModel):
    song_id: int


class CreatePlaylistRequest(BaseModel):
    user_id: str
    playlist_name: str
    playlist_items: List[PlaylistItem]


# playlist ittem fetcher
def get_song_playlist_items_by_id(conn, playlist_id: int):
    result = []
    try:
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                """
                SELECT s.*
                FROM user_playlist p
                JOIN LATERAL jsonb_array_elements(p.playlist_items) AS item ON TRUE
                JOIN songs s ON (item->>'song_id')::INT = s.id
                WHERE p.id = %s
                """,
                (playlist_id,),
            )
            songs = cursor.fetchall()
            for song in songs:
                result.append(
                    {
                        "song_id": song["id"],
                        "song": song["song"],
                        "artist": song["artist"],
                        "full_lyric": "",
                        "dominants": song["dominants"],
                        "tags": song["tags"],
                        "genre": song["genre"],
                    }
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    return result


@app.get("/api/get-user-playlist")
async def get_user_playlist(user_id: str):
    """
    Get all playlists for a specific user
    """

    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)

        try:
            # Query to get all playlists for the user
            cursor.execute(
                """
                SELECT id, user_id, playlist_name, playlist_items, created_at
                FROM user_playlist
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )

            playlists = []
            for row in cursor.fetchall():
                playlists.append(
                    {
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "playlist_name": row["playlist_name"],
                        "playlist_items": row["playlist_items"],
                    }
                )

            return {"playlists": playlists}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.delete("/api/delete-user-playlist")
async def delete_user_playlist(user_id: str, playlist_id: int):
    """
    Delete a playlist for a user and return remaining playlists
    """

    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)

        try:
            # Check if playlist exists and belongs to the user
            cursor.execute(
                """
                SELECT id FROM user_playlist 
                WHERE id = %s AND user_id = %s
                """,
                (playlist_id, user_id),
            )

            playlist = cursor.fetchone()
            if not playlist:
                raise HTTPException(
                    status_code=404,
                    detail="Playlist not found or doesn't belong to the user",
                )

            # Delete the playlist
            cursor.execute("DELETE FROM user_playlist WHERE id = %s", (playlist_id,))

            conn.commit()

            # Get remaining playlists for the user
            cursor.execute(
                """
                SELECT id, user_id, playlist_name, playlist_items
                FROM user_playlist
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )

            playlists = []
            for row in cursor.fetchall():
                playlists.append(
                    {
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "playlist_name": row["playlist_name"],
                        "playlist_items": row["playlist_items"],
                    }
                )

            return {"playlists": playlists}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.post("/api/create-user-playlist")
async def create_user_playlist(request: CreatePlaylistRequest):
    """
    Create a new playlist for a user
    """

    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)

        try:
            # Convert playlist items to JSON string
            playlist_items_json = Json(
                [{"song_id": item.song_id} for item in request.playlist_items]
            )

            # Insert the new playlist
            cursor.execute(
                """
                INSERT INTO user_playlist (user_id, playlist_name, playlist_items)
                VALUES (%s, %s, %s)
                RETURNING id, created_at
                """,
                (request.user_id, request.playlist_name, playlist_items_json),
            )

            result = cursor.fetchone()
            conn.commit()

            playlist = {
                "id": result["id"],
                "user_id": request.user_id,
                "playlist_name": request.playlist_name,
                "playlist_items": request.playlist_items,
                "created_at": (
                    result["created_at"].isoformat() if result["created_at"] else None
                ),
            }

            new_playlist_items = get_song_playlist_items_by_id(conn, result["id"])

            cursor.execute(
                """
            SELECT id, user_id, playlist_name, playlist_items, created_at
            FROM user_playlist
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
                (request.user_id,),
            )

            playlists = []

            for row in cursor.fetchall():
                playlists.append(
                    {
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "playlist_name": row["playlist_name"],
                        "playlist_items": row["playlist_items"],
                    }
                )

            new_item = {"playlist": playlist, "items": new_playlist_items}

            return {"playlists": playlists, "new_item": new_item}

        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.post("/api/update-user-playlist")
async def update_user_playlist(playlist_id: int, song_id: int, action: str = "add"):
    """
    Add or remove a song from an existing playlist
    """

    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)

        try:
            # First get the current playlist items
            cursor.execute("SELECT * FROM user_playlist WHERE id = %s", (playlist_id,))

            playlist = cursor.fetchone()
            if not playlist:
                raise HTTPException(status_code=404, detail="Playlist not found")
            print(playlist)

            playlist_items = playlist["playlist_items"]
            print(playlist_items)

            if action == "add":
                # Check if song already exists in playlist
                song_exists = any(
                    item.get("song_id") == song_id for item in playlist_items
                )

                if not song_exists:
                    # Add the new song to the playlist items
                    playlist_items.append({"song_id": song_id})

                    # Update the playlist with the new items
                    cursor.execute(
                        "UPDATE user_playlist SET playlist_items = %s::jsonb WHERE id = %s RETURNING id",
                        (Json(playlist_items), playlist_id),
                    )

                    conn.commit()

                    pl = {
                        "id": playlist["id"],
                        "user_id": playlist["user_id"],
                        "playlist_name": playlist["playlist_name"],
                        "playlist_items": playlist_items,
                    }

                    items = get_song_playlist_items_by_id(conn, playlist_id)

                    return {
                        "message": "Song added to playlist",
                        "playlist": pl,
                        "items": items,
                    }
                else:
                    return {
                        "message": "Song already exists in playlist",
                        "playlist": {},
                        "items": [],
                    }
            elif action == "remove":
                # Filter out the song to remove
                new_playlist_items = [
                    item for item in playlist_items if item.get("song_id") != song_id
                ]

                if len(new_playlist_items) < len(playlist_items):
                    # Update the playlist with the filtered items
                    cursor.execute(
                        "UPDATE user_playlist SET playlist_items = %s::jsonb WHERE id = %s RETURNING id",
                        (json.dumps(new_playlist_items), playlist_id),
                    )

                    conn.commit()

                    pl = {
                        "id": playlist["id"],
                        "user_id": playlist["user_id"],
                        "playlist_name": playlist["playlist_name"],
                        "playlist_items": new_playlist_items,
                    }

                    items = get_song_playlist_items_by_id(conn, playlist_id)

                    return {
                        "message": "Song removed from playlist",
                        "playlist": pl,
                        "items": items,
                    }

                else:
                    return {
                        "message": "Song not found in playlist",
                        "playlist_id": {},
                        "song_id": [],
                    }
            else:
                raise HTTPException(
                    status_code=400, detail="Invalid action. Use 'add' or 'remove'."
                )

        except HTTPException as e:
            raise e
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/get-song-playlist-by-id")
async def get_song_playlist_by_id(id: int):
    """
    Get song by id
    """

    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)

        try:
            # Query to get all playlists for the user
            cursor.execute(
                """
                SELECT id, user_id, playlist_name, playlist_items
                FROM user_playlist
                WHERE id = %s
                """,
                (id,),
            )

            playlist = cursor.fetchone()
            if not playlist:
                raise HTTPException(status_code=404, detail="Song not found")
            selected_playlist = {
                "id": playlist["id"],
                "user_id": playlist["user_id"],
                "playlist_name": playlist["playlist_name"],
                "playlist_items": playlist["playlist_items"],
            }

            items = get_song_playlist_items_by_id(conn, playlist["id"])

            return {"playlist": selected_playlist, "items": items}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/get-song-by-id")
async def get_song_by_id(id: int):
    """
    Get song by id
    """

    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)

        try:
            # Query to get all playlists for the user
            cursor.execute(
                """
                SELECT id, song, artist, full_lyric, dominants, tags, genre
                FROM songs
                WHERE id = %s
                """,
                (id,),
            )

            song = cursor.fetchone()
            if not song:
                raise HTTPException(status_code=404, detail="Song not found")

            song_item = {
                "song_id": song["id"],
                "song": song["song"],
                "artist": song["artist"],
                "full_lyric": "",
                "dominants": song["dominants"],
                "tags": song["tags"],
                "genre": song["genre"],
            }

            return song_item

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/get-playlist-by-playlist-id")
async def get_user_playlist(id: int):

    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)

        try:
            cursor.execute(
                """
                SELECT id, user_id, playlist_name, playlist_items
                FROM user_playlist
                WHERE id = %s
                ORDER BY created_at DESC
                """,
                (id,),
            )

            playlist = cursor.fetchone()
            if not playlist:
                raise HTTPException(status_code=404, detail="Playlist not found")

            selected_playlist = {
                "id": playlist["id"],
                "user_id": playlist["user_id"],
                "playlist_name": playlist["playlist_name"],
                "playlist_items": playlist["playlist_items"],
            }

            items = get_song_playlist_items_by_id(conn, playlist["id"])

            return {"playlist": selected_playlist, "items": items}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)
