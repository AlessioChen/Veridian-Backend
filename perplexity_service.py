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
                    "You are a helpful career coach that retrieves sources to help individuals advance their careers"
                ),
            },
            {
                "role": "user",
                "content": (
                    query
                ),
            },
        ]

        # chat completion without streaming
        response = self.client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=messages,
        )
        print(response)
