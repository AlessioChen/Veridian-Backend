import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


class GroqServices:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def speech_to_text(self, filename):
        with open(filename, 'rb') as file:
            file_content = file.read()

            translation = self.client.audio.translations.create(
                file=(filename, file_content),  # Required audio file
                model="whisper-large-v3", 
                prompt="Transcribe the following audio into text", 
                response_format="json",  
                temperature=0.0  
            )

        return translation.text
