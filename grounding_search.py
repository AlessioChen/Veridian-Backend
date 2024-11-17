# Grounding search using Perplexity's API
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class PerplexityGenericSearch:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")

    def search(self, query: str):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search assistant focused on providing factual information. "
                    "Return only relevant facts and information from reliable sources. "
                    "Format your response in a clear, concise manner. "
                    "Always include sources or citations when available. "
                    "Do not include personal opinions or speculative content."
                )
            },
            {
                "role": "user",
                "content": f"Provide factual information about: {query}",
            },
        ]

        response = self.client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=messages,
            temperature=0,  # Set to 0 for maximum factuality
            presence_penalty=1,  # Removed penalties to focus on direct answers
        
            stream=False
        )
        
        return response.choices[0].message.content 