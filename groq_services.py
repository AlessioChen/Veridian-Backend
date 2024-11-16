import os
from groq import Groq
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)


class GroqServices:
    async def process_audio_to_text(file_data: bytes, filename: str) -> str:
        """
        Process MP3 data with Groq's Whisper API
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file.write(file_data)

        with open(filename, "rb") as file:
            # Create a translation of the audio file
            translation = groq_client.audio.translations.create(
                file=(filename, file.read()),  # Required audio file
                model="whisper-large-v3",  # Required model to use for translation
                prompt="Transcribe the following audio into text",  # Optional
                response_format="json",  # Optional
                temperature=0.0  # Optional
            )

            return translation.text
