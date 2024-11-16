import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class GroqServices(Groq):
    def speech_to_text(groq_client, file):
        with open(filename, "rb") as file:
            # Create a translation of the audio file
            translation = groq_client.audio.translations.create(
                file=(filename, file.read()),  # Required audio file
                model="whisper-large-v3",  # Required model to use for translation
                prompt="Specify context or spelling",  # Optional
                response_format="json",  # Optional
                temperature=0.0  # Optional
            )
    
            return translation.text


if __name__ == '__main__':
    api_key = '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'
    client = Groq(api_key=api_key)

    filename = os.path.dirname(__file__) + "/test.mp3"  # Replace with your audio file!
    text = speech_to_text(client, filename)
    print(text)
