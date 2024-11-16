import os
from groq import Groq
import tempfile
from dotenv import load_dotenv


load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)





class GroqServices:

    def speech_to_text(filename):
        with open(filename, "rb") as file:

            translation = groq_client.audio.translations.create(
                file=(filename, file.read()), # Required audio file
                model="whisper-large-v3", 
                prompt="Transcribe the following audio into text", 
                response_format="json",  
                temperature=0.0  
            )

        return translation.text
