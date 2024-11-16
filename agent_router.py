from enum import Enum
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
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
        self.llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API'),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        self.output_parser = CommaSeparatedListOutputParser()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent router that determines which specialized agent should handle user requests.
            Consider the user's message and context to make your decision.
            - Use career_agent for career advice, job searching, and professional development
            - Use general_agent for all other topics and general conversation
            
            Respond with exactly two values separated by a comma:
            1. Either 'career_agent' or 'general_agent'
            2. Your reason for the choice"""),
            ("human", "{message}")
        ])
        
        self.chain = self.prompt | self.llm | self.output_parser

    async def route(self, request: RouterRequest):
        try:
            result = await self.chain.ainvoke({"message": request.message})
            return result[0], result[1]
        except Exception as e:
            return "general_agent", f"Defaulting to general agent due to error: {str(e)}" 