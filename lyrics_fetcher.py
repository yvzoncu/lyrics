import lyricsgenius
import re
from dataclasses import dataclass



API_KEY = '6HDti8ZGSyFbR-pwUPFfi_1mCCgxIeziCo8E1vnyZw-4GARuCOhW29Eb638ucO5s'

def clean_text(text, default_chars=r"[.,!?\"'();::/\\-]"):
    escaped_custom_chars = re.escape(default_chars)
    pattern = f"[{escaped_custom_chars}]"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text

@dataclass
class LyricsFetcher:
    def __init__(self):
        self.api_key = API_KEY
        if not self.api_key:
            raise ValueError("Missing GENIUS_API_KEY in environment variables")
        
        # Configure API client
        self.genius = lyricsgenius.Genius(
            self.api_key,
            timeout=15,
            retries=3,
            verbose=False,
            skip_non_songs=True,
            remove_section_headers=True
        )

    def get_lyrics(self, title, artist):
        """Fetch lyrics with proper error handling and caching"""
        try:
            song = self.genius.search_song(title, artist)
            
            if not song:
                return None
            

            
            print("Image URL:", song.song_art_image_url)

            return {
                "metadata": {
                    "title": song.title,
                    "artist": song.artist,
                    "url": song.url,
                    
                   
                   
                },
                "lyrics": song.lyrics,
                "attribution": {
                    "source": "Genius API",
                    "terms": "https://genius.com/static/terms"
                }
            }
            
        except Exception as e:
            return None


    def get_contributors(self, lyrics):
    # Use regular expression to find the contributors count
        contributors_match = re.search(r'(\d+) Contributors', lyrics)

        if contributors_match:
            contributors_count = contributors_match.group(1)
            return int(contributors_count)
        else:
            return 0   

        
    def extract_lyrics(self, lyrics):
             
        lines = lyrics.split('\n')
        
        lyrics_lines = []
        for line in lines:
            if not any(keyword in line for keyword in ["Contributors", "Translations", "Español", "Deutsch", "Українська", "Norsk", "Français", "Lyrics"]):
                    # Remove any leading or trailing whitespace
                    line = line.strip()
                    # Skip empty lines
                    if not line:
                        continue
                    # Skip lines that are just square brackets
                    if line == "[" or line == "]":
                        continue
                    # Skip lines that are just numbers
                    if line.isdigit():
                        continue
                    # Skip lines that are just "Embed"
                    if line == "Embed":
                        continue
                    line = clean_text(line)

                    if line not in lyrics_lines:
                        lyrics_lines.append(line)
                        
        return lyrics_lines 
                

