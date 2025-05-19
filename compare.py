import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pydantic import BaseModel
import os
import psycopg2
from psycopg2.extras import DictCursor
from collections import defaultdict
import pickle
import json
from evaluator import tag_finder
from lyrics_fetcher import LyricsFetcher


class SongRequest(BaseModel):
    song: str
    artist: str
    ulr: str
    contributor: int


model_name = "AnkitAI/deberta-v3-small-base-emotions-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = model.config.id2label

model = model.to("cpu")


def analyze_song_emotions(lyrics_sections):
    emotion_results = {}

    for section, text in lyrics_sections.items():
        emotion_probs = get_emotion(text)
        emotion_results[section.strip()] = emotion_probs[1]

    return emotion_results


def get_emotion(text):
    # Tokenize input
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    # Extract prediction
    predicted_idx = torch.argmax(probs, dim=1).item()
    predicted_emotion = labels[predicted_idx]

    # Format and filter probabilities
    prob_dict = {
        labels[i]: float(probs[0][i])
        for i in range(len(labels))
        if float(probs[0][i]) > 0.01
    }

    # Sort by value descending
    sorted_probs = dict(
        sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    )

    return predicted_emotion, sorted_probs


def text_to_emotion(text):
    # Tokenize input with a smaller max_length
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    # Get model output with a batch size of 1
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    emotion_vector = [float(probs[0][i]) for i in range(len(labels))]

    return emotion_vector


def get_lyrics_sections(genius_url):
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(genius_url, headers=headers)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "html.parser")

    # Lyrics are often inside a <div> with data-lyrics-container="true"
    lyrics_divs = soup.find_all("div", {"data-lyrics-container": "true"})

    full_lyrics = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])

    # Split by section headers like [Chorus], [Verse 1], etc.
    sections = re.split(r"\[([^\[\]]+)\]", full_lyrics)

    structured = {}
    current_section = "Intro"
    for i in range(1, len(sections), 2):
        current_section = sections[i].strip()
        lyrics = sections[i + 1].strip()
        structured[current_section] = lyrics

    return structured


def cleean_lyris(lyrics_parts):
    full_lyrics = ""
    for section, text in lyrics_parts.items():
        section_clean = section.replace("\n", " ").strip()
        full_lyrics += text
    return full_lyrics


# Environment variables
DB_HOST = os.getenv("DB_HOST", "moodify.cje0wa8qioij.eu-north-1.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "lyricsdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=DictCursor,
    )


def check_song_exists(genius_id, cursor):
    try:
        cursor.execute(
            """
            SELECT genius_id, dominant_emotions, tags FROM songs
            WHERE genius_id = %s
            """,
            (genius_id,),
        )
        existing = cursor.fetchone()
        print("Query result:", existing)

        if existing:
            return False  # Song exists
        return True  # Song does not exist
    except Exception as e:
        print(f"An error occurred: {e}")
        return False  # On error, default to "exists" to avoid duplicates


def create_song_meta(song, artist, full_lyric, genius_id, genre, cursor, conn):
    try:
        # Step 1: Check if song already exists
        cursor.execute(
            """
            INSERT INTO songs (song, artist, full_lyric, genius_id, genre )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                song,
                artist,
                full_lyric,
                genius_id,
                genre,
            ),
        )
        song_id = cursor.fetchone()["id"]
        conn.commit()
        return song_id
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_lyrics(lyric_sections, emotion_results, song_id, cursor, conn):
    emotion_totals = defaultdict(float)

    for section, emotion_dict in emotion_results.items():
        text = lyric_sections.get(section, "").strip()
        if not text:
            continue

        # Sum up emotion probabilities
        for emotion, value in emotion_dict.items():
            emotion_totals[emotion] += value

        emotion_vector = list(emotion_dict.values())

        vector_bytes = psycopg2.Binary(pickle.dumps(emotion_vector))

        cursor.execute(
            """
                        INSERT INTO lyrics (
                            song_id, lyric, embedding, section,
                            sadness, joy, love, anger, fear, surprise
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
            (
                song_id,
                text,
                vector_bytes,
                section,
                emotion_dict.get("sadness", 0.0),
                emotion_dict.get("joy", 0.0),
                emotion_dict.get("love", 0.0),
                emotion_dict.get("anger", 0.0),
                emotion_dict.get("fear", 0.0),
                emotion_dict.get("surprise", 0.0),
            ),
        )
        lyric_id = cursor.fetchone()["id"]
        conn.commit()

    # Determine the most dominant emotion(s)
    emotion_totals = defaultdict(float)

    for section, emotions in emotion_results.items():
        for emotion, value in emotions.items():
            emotion_totals[emotion] += value

        final_totals = dict(emotion_totals)
        top_3_emotions = sorted(final_totals.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]
        top_3_formatted = [
            {"emotion": emotion, "value": round(value, 6)}
            for emotion, value in top_3_emotions
        ]

        dominant_json = json.dumps(top_3_formatted)

        return dominant_json


def insert_dominant_emotions(de, tags, song_id, cursor, conn):
    try:
        # Step 1: Check if song already exists
        cursor.execute(
            """
                    UPDATE songs
                    SET 
                        dominant_emotions = %s,
                        tags = %s
                    WHERE id = %s
                    """,
            (
                de,
                tags,
                song_id,
            ),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


top_songs = [
    {
        "artist": "Eminem",
        "song": "Houdini",
        "genre": ["Hip-Hop", "Rap"],
        "genius_url": "https://genius.com/Eminem-houdini-lyrics",
    },
    {
        "artist": "Taylor Swift",
        "song": "Fortnight",
        "genre": ["Pop", "Alternative"],
        "genius_url": "https://genius.com/Taylor-swift-fortnight-lyrics",
    },
    {
        "artist": "Hozier",
        "song": "Too Sweet",
        "genre": ["Indie Folk", "Blues Rock"],
        "genius_url": "https://genius.com/Hozier-too-sweet-lyrics",
    },
    {
        "artist": "Tommy Richman",
        "song": "Million Dollar Baby",
        "genre": ["R&B", "Pop"],
        "genius_url": "https://genius.com/Tommy-richman-million-dollar-baby-lyrics",
    },
    {
        "artist": "Teddy Swims",
        "song": "Lose Control",
        "genre": ["Soul", "Pop"],
        "genius_url": "https://genius.com/Teddy-swims-lose-control-lyrics",
    },
    {
        "artist": "Sabrina Carpenter",
        "song": "Espresso",
        "genre": ["Pop", "Dance-Pop"],
        "genius_url": "https://genius.com/Sabrina-carpenter-espresso-lyrics",
    },
    {
        "artist": "Sabrina Carpenter",
        "song": "Please Please Please",
        "genre": ["Pop"],
        "genius_url": "https://genius.com/Sabrina-carpenter-please-please-please-lyrics",
    },
    {
        "artist": "Shaboozey",
        "song": "A Bar Song (Tipsy)",
        "genre": ["Country Rap", "Country Pop"],
        "genius_url": "https://genius.com/Shaboozey-a-bar-song-tipsy-lyrics",
    },
    {
        "artist": "Benson Boone",
        "song": "Beautiful Things",
        "genre": ["Pop", "Ballad"],
        "genius_url": "https://genius.com/Benson-boone-beautiful-things-lyrics",
    },
    {
        "artist": "Post Malone",
        "song": "I Had Some Help",
        "genre": ["Country-Pop", "Hip-Hop"],
        "genius_url": "https://genius.com/Post-malone-i-had-some-help-lyrics",
    },
    {
        "artist": "Ariana Grande",
        "song": "We Can’t Be Friends",
        "genre": ["Pop", "R&B"],
        "genius_url": "https://genius.com/Ariana-grande-we-cant-be-friends-wait-for-your-love-lyrics",
    },
    {
        "artist": "Dua Lipa",
        "song": "Illusion",
        "genre": ["Pop", "Disco-Pop"],
        "genius_url": "https://genius.com/Dua-lipa-illusion-lyrics",
    },
    {
        "artist": "Kendrick Lamar",
        "song": "Not Like Us",
        "genre": ["West Coast Hip-Hop", "Diss Track"],
        "genius_url": "https://genius.com/Kendrick-lamar-not-like-us-lyrics",
    },
    {
        "artist": "Future, Metro Boomin & Kendrick Lamar",
        "song": "Like That",
        "genre": ["Trap", "Hip-Hop"],
        "genius_url": "https://genius.com/Future-metro-boomin-and-kendrick-lamar-like-that-lyrics",
    },
    {
        "artist": "Beyoncé",
        "song": "Texas Hold 'Em",
        "genre": ["Country", "Pop"],
        "genius_url": "https://genius.com/Beyonce-texas-hold-em-lyrics",
    },
    {
        "artist": "Noah Kahan",
        "song": "Stick Season",
        "genre": ["Folk-Pop", "Indie Folk"],
        "genius_url": "https://genius.com/Noah-kahan-stick-season-lyrics",
    },
    {
        "artist": "FloyyMenor & Cris Mj",
        "song": "Gata Only",
        "genre": ["Reggaeton", "Latin Trap"],
        "genius_url": "https://genius.com/Floyymenor-and-cris-mj-gata-only-lyrics",
    },
    {
        "artist": "Xavi",
        "song": "La Diabla",
        "genre": ["Regional Mexican", "Corrido Tumbado"],
        "genius_url": "https://genius.com/Xavi-la-diabla-lyrics",
    },
    {
        "artist": "Ivan Cornejo",
        "song": "Perdoname",
        "genre": ["Sad Sierreño", "Regional Mexican"],
        "genius_url": "https://genius.com/Ivan-cornejo-perdoname-lyrics",
    },
    {
        "artist": "Jack Harlow",
        "song": "Lovin On Me",
        "genre": ["Pop-Rap"],
        "genius_url": "https://genius.com/Jack-harlow-lovin-on-me-lyrics",
    },
    {
        "artist": "Djo",
        "song": "End of Beginning",
        "genre": ["Indie Rock", "Psychedelic Pop"],
        "genius_url": "https://genius.com/Djo-end-of-beginning-lyrics",
    },
    {
        "artist": "Kanye West & Ty Dolla $ign",
        "song": "Carnival",
        "genre": ["Hip-Hop", "Gospel Rap"],
        "genius_url": "https://genius.com/Kanye-west-and-ty-dolla-sign-carnival-lyrics",
    },
    {
        "artist": "SZA",
        "song": "Saturn",
        "genre": ["Alternative R&B"],
        "genius_url": "https://genius.com/Sza-saturn-lyrics",
    },
    {
        "artist": "Ariana Grande",
        "song": "Yes, And?",
        "genre": ["Pop", "House"],
        "genius_url": "https://genius.com/Ariana-grande-yes-and-lyrics",
    },
    {
        "artist": "Tyla",
        "song": "Water",
        "genre": ["Afropop", "Amapiano"],
        "genius_url": "https://genius.com/Tyla-water-lyrics",
    },
    {
        "artist": "Sophie Ellis-Bextor",
        "song": "Murder on the Dancefloor",
        "genre": ["Disco-Pop", "Nu-Disco"],
        "genius_url": "https://genius.com/Sophie-ellis-bextor-murder-on-the-dancefloor-lyrics",
    },
    {
        "artist": "Dua Lipa",
        "song": "Training Season",
        "genre": ["Pop", "Disco-Pop"],
        "genius_url": "https://genius.com/Dua-lipa-training-season-lyrics",
    },
    {
        "artist": "Natasha Bedingfield",
        "song": "Unwritten",
        "genre": ["Pop"],
        "genius_url": "https://genius.com/Natasha-bedingfield-unwritten-lyrics",
    },
    {
        "artist": "Taylor Swift",
        "song": "Cruel Summer",
        "genre": ["Synth-Pop"],
        "genius_url": "https://genius.com/Taylor-swift-cruel-summer-lyrics",
    },
    {
        "artist": "Doja Cat",
        "song": "Paint the Town Red",
        "genre": ["Hip-Hop", "Pop-Rap"],
        "genius_url": "https://genius.com/Doja-cat-paint-the-town-red-lyrics",
    },
    {
        "artist": "Sabrina Carpenter",
        "song": "Feather",
        "genre": ["Pop", "Dance-Pop"],
        "genius_url": "https://genius.com/Sabrina-carpenter-feather-lyrics",
    },
    {
        "artist": "The Weeknd, Madonna & Playboi Carti",
        "song": "Popular",
        "genre": ["Pop", "Hip-Hop"],
        "genius_url": "https://genius.com/The-weeknd-madonna-and-playboi-carti-popular-lyrics",
    },
    {
        "artist": "Dua Lipa",
        "song": "Dance The Night",
        "genre": ["Disco-Pop"],
        "genius_url": "https://genius.com/Dua-lipa-dance-the-night-lyrics",
    },
    {
        "artist": "Jelly Roll",
        "song": "Save Me",
        "genre": ["Country-Rock"],
        "genius_url": "https://genius.com/Jelly-roll-save-me-lyrics",
    },
    {
        "artist": "Morgan Wallen",
        "song": "Thinkin' Bout Me",
        "genre": ["Country-Pop"],
        "genius_url": "https://genius.com/Morgan-wallen-thinkin-bout-me-lyrics",
    },
    {
        "artist": "The Weeknd, Jennie & Lily-Rose Depp",
        "song": "One of the Girls",
        "genre": ["Dark Pop", "R&B"],
        "genius_url": "https://genius.com/The-weeknd-jennie-and-lily-rose-depp-one-of-the-girls-lyrics",
    },
    {
        "artist": "Morgan Wallen",
        "song": "Last Night",
        "genre": ["Country-Pop"],
        "genius_url": "https://genius.com/Morgan-wallen-last-night-lyrics",
    },
    {
        "artist": "Luke Combs",
        "song": "Fast Car",
        "genre": ["Country"],
        "genius_url": "https://genius.com/Luke-combs-fast-car-lyrics",
    },
    {
        "artist": "Miley Cyrus",
        "song": "Flowers",
        "genre": ["Pop"],
        "genius_url": "https://genius.com/Miley-cyrus-flowers-lyrics",
    },
    {
        "artist": "SZA",
        "song": "Kill Bill",
        "genre": ["R&B", "Alternative R&B"],
        "genius_url": "https://genius.com/Sza-kill-bill-lyrics",
    },
    {
        "artist": "Rema & Selena Gomez",
        "song": "Calm Down",
        "genre": ["Afrobeats", "Pop"],
        "genius_url": "https://genius.com/Rema-and-selena-gomez-calm-down-lyrics",
    },
    {
        "artist": "Drake",
        "song": "IDGAF (feat. Yeat)",
        "genre": ["Trap", "Hip-Hop"],
        "genius_url": "https://genius.com/Drake-idgaf-lyrics",
    },
    {
        "artist": "Jung Kook",
        "song": "Seven (feat. Latto)",
        "genre": ["Pop", "K-Pop"],
        "genius_url": "https://genius.com/Jung-kook-seven-lyrics",
    },
    {
        "artist": "Kenya Grace",
        "song": "Strangers",
        "genre": ["Drum & Bass", "Pop"],
        "genius_url": "https://genius.com/Kenya-grace-strangers-lyrics",
    },
    {
        "artist": "Xavi",
        "song": "La Víctima",
        "genre": ["Regional Mexican"],
        "genius_url": "https://genius.com/Xavi-la-victima-lyrics",
    },
    {
        "artist": "Anne-Marie & KSI",
        "song": "Don't Play",
        "genre": ["Pop", "UK Pop"],
        "genius_url": "https://genius.com/Anne-marie-and-ksi-dont-play-lyrics",
    },
    {
        "artist": "Myke Towers",
        "song": "Lala",
        "genre": ["Reggaeton"],
        "genius_url": "https://genius.com/Myke-towers-lala-lyrics",
    },
    {
        "artist": "Harry Styles",
        "song": "As It Was",
        "genre": ["Pop", "Synth-Pop"],
        "genius_url": "https://genius.com/Harry-styles-as-it-was-lyrics",
    },
    {
        "artist": "SZA",
        "song": "Snooze",
        "genre": ["R&B"],
        "genius_url": "https://genius.com/Sza-snooze-lyrics",
    },
    {
        "artist": "The Weeknd",
        "song": "Blinding Lights",
        "genre": ["Synthwave", "Pop"],
        "genius_url": "https://genius.com/The-weeknd-blinding-lights-lyrics",
    },
]

top_songs1 = [
    {
        "artist": "Eminem",
        "song": "Houdini",
        "genre": ["Hip-Hop", "Rap"],
        "genius_url": "https://genius.com/Eminem-houdini-lyrics",
    },
]


def list_importer(slist):
    conn = get_db_connection()
    cursor = conn.cursor()

    lf = LyricsFetcher()

    for s in slist:
        artist = s.get("artist")
        song = s.get("song")
        genre = s.get("genre")

        attr = lf.get_lyrics(song, artist)

        if not attr:
            continue

        meta = attr.get("metadata")
        print(type(meta))

        genius_id = str(meta["id"])
        title = meta["title"]
        artist = meta["artist"]
        url = meta["url"]
        print(f"attr: {genius_id}{title}")

        check_song = check_song_exists(genius_id, cursor)
        print(check_song)
        if not check_song:
            continue

        lyric_sections = get_lyrics_sections(url)
        if not lyric_sections:
            continue
        full_lyrics = cleean_lyris(lyric_sections)

        emotion_results = analyze_song_emotions(lyric_sections)
        print(emotion_results)

        song_id = create_song_meta(
            title, artist, full_lyrics, genius_id, json.dumps(genre), cursor, conn
        )
        if not song_id:
            print("no song id")
            continue

        dominant_emotions = process_lyrics(
            lyric_sections, emotion_results, song_id, cursor, conn
        )
        print(dominant_emotions)

        song_tags = tag_finder(full_lyrics, title, artist)
        print(song_tags)

        update = insert_dominant_emotions(
            json.dumps(dominant_emotions), song_tags, song_id, cursor, conn
        )

    cursor.close()
    conn.close()


def updater():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT id, dominant_emotions
            FROM songs;
            """
        )
        existing = cursor.fetchall()

        for song_id, dominant_emotions_str in existing:
            # Parse the dominant_emotions string into JSON
            emotions_list = json.loads(dominant_emotions_str)  # List of dicts

            # Reformat the emotions to a list of single-key dicts with emotion:value
            new_format = [{item["emotion"]: item["value"]} for item in emotions_list]

            # Convert back to JSON string for updating the dominants column
            new_format_json = json.dumps(new_format)

            # Update the dominants column for this song
            cursor.execute(
                """
                UPDATE songs
                SET dominants = %s
                WHERE id = %s;
                """,
                (new_format_json, song_id),
            )

        conn.commit()

    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
        return False

    finally:
        cursor.close()
        conn.close()

    return True
