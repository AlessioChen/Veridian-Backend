from enum import Enum
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class AgentType(str, Enum):
    CAREER = "career_agent"
    GENERAL = "general_agent"

class RouterRequest(BaseModel):
    message: str

class Router:
    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API'))

    def route(self, request: RouterRequest):
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an intelligent router that determines which specialized agent should handle user requests.
                        Consider the user's message and context to make your decision.
                        - Use career_agent for career advice, job searching, and professional development
                        - Use general_agent for all other topics and general conversation"""
                    },
                    {
                        "role": "user",
                        "content": f"Route this message: {request.message}\nRespond with either 'career_agent' or 'general_agent' followed by a comma and then your reason."
                    }
                ],
                temperature=0,
                max_tokens=100,
                top_p=1,
            )
            
            result = response.choices[0].message.content
            agent_type, reason = result.split(',', 1)
            return agent_type.strip(), reason.strip()
            
        except Exception as e:
            return "general_agent", f"Defaulting to general agent due to error: {str(e)}" 