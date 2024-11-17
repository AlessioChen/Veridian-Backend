import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class PerplexityService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")

    def chat_request(self, query: str):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a URL retrieval system. Provide relevant career-related URLs in the following JSON format only:"
                    "\n{\"urls\": [{\"title\": \"Article Title\", \"url\": \"https://example.com\", \"description\": \"Brief description\"}]}"
                    "\nReturn raw JSON only, without code blocks or markdown formatting."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        response = self.client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=messages,
            temperature=0,
            presence_penalty=0.5,  # Adjusted presence penalty for better diversity
            frequency_penalty=1,  # Adjusted frequency penalty to reduce repetition
            stream=False
        )
        
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return content
