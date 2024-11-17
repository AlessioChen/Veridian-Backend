import os
from typing import AsyncGenerator, Dict, Any, Annotated
from enum import Enum
from pydantic import BaseModel
from fastapi import HTTPException
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Sequence, Union, cast
from langgraph.graph.message import add_messages

class AgentType(str, Enum):
    CAREER = "career"
    GENERAL = "general"
    RESUME = "resume"
    INTERVIEW = "interview"
    SKILLS = "skills"
    NETWORKING = "networking"
    JOB_SEARCH = "job_search"

class ChatRequest(BaseModel):
    message: str

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_type: str
    history: list

class LLMService:
    def __init__(self):
        print("Initializing LLM Service")  # Debug
        # Initialize LLM configurations with streaming enabled
        self.router_llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API'),
            model_name="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=256,
            streaming=True
        )
        
        self.agent_llm = ChatGroq(
            groq_api_key=os.getenv('GROQ_API'),
            model_name="llama-3.2-90b-text-preview",
            temperature=0.2,
            max_tokens=1024,
            streaming=True
        )
        
        # Initialize prompts
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent router that determines which specialized agent should handle user requests.
            - Use 'career' for general career advice and professional development
            - Use 'resume' for resume and cover letter optimization
            - Use 'interview' for interview preparation and practice
            - Use 'skills' for skill gap analysis and learning recommendations
            - Use 'networking' for networking strategies and professional connections
            - Use 'job_search' for job search assistance and application help
            - Use 'general' for all other topics and general conversation
            
            Respond with only one word from the options above."""),
            ("human", "{message}")
        ])
        
        self.agent_prompts = {
        AgentType.CAREER: ChatPromptTemplate.from_messages([
            ("system", """You are a genius UK-based career advisor helping users transition careers. 
            Your responses should not include any markdown formatting.
            - Match their current skills to relevant UK jobs.
            - Reference UK-specific job boards (e.g., Indeed, Reed, TotalJobs) and funding schemes (e.g., National Careers Service, Skills Bootcamps).
            - Provide concise, actionable steps and explain your reasoning clearly."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{message}")
        ]),
        AgentType.GENERAL: ChatPromptTemplate.from_messages([
            ("system", """You are a UK-focused career assistant. Ensure these areas are covered:
            Your responses should not include any markdown formatting.
            - Current job, salary, work experience, skills, industry, education, location, time constraints, and goals.
            - Prompt the user for missing information.
            - Give concise, professional answers tailored to UK users."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{message}")
        ]),
        AgentType.RESUME: ChatPromptTemplate.from_messages([
            ("system", """You are an expert in UK CV and cover letter optimisation. 
            Your responses should not include any markdown formatting.
            - Highlight transferable skills and use UK job market keywords.
            - Focus on achievements and maintain ATS-friendly formatting.
            - Provide specific, actionable edits."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{message}")
        ]),
        AgentType.INTERVIEW: ChatPromptTemplate.from_messages([
            ("system", """You are an interview preparation expert for the UK job market.
            Your responses should not include any markdown formatting.
            - Conduct mock interviews.
            - Provide common and industry-specific UK interview questions.
            - Offer constructive feedback and share best practices."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{message}")
        ]),
        AgentType.SKILLS: ChatPromptTemplate.from_messages([
            ("system", """You are a UK skill development advisor.
            Your responses should not include any markdown formatting.
            - Identify skill gaps for target jobs.
            - Recommend relevant courses, e.g., Skills Bootcamps, Open University.
            - Suggest practical exercises and learning paths."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{message}")
        ]),
        AgentType.NETWORKING: ChatPromptTemplate.from_messages([
            ("system", """You are a UK professional networking advisor.
            Your responses should not include any markdown formatting.
            - Suggest strategies for LinkedIn UK, local meetups, and industry events.
            - Help optimise LinkedIn profiles.
            - Provide outreach templates tailored to UK industries."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{message}")
        ]),
        AgentType.JOB_SEARCH: ChatPromptTemplate.from_messages([
            ("system", """You are a UK job search expert.
            Your responses should not include any markdown formatting.
            - Help users find jobs on UK boards (e.g., Reed, TotalJobs, GOV.UK Find a Job).
            - Tailor applications for specific roles.
            - Suggest tracking methods and job search strategies."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{message}")
        ])
    }

        
        # Initialize the graph
        self.workflow = self._create_graph()
        
        # Add conversation history storage
        self.conversation_history = {}  # Store history by user_id

    async def route_message(self, state: ChatState) -> ChatState:
        print("Routing message")  # Debug
        messages = state["messages"]
        last_message = cast(HumanMessage, messages[-1])
        
        chain = self.router_prompt | self.router_llm
        # Add debug logging for router payload
        router_payload = {"message": last_message.content}
        print(f"Router API Payload: {router_payload}")  # Debug
        result = await chain.ainvoke(router_payload)
        print(f"Routed to: {result.content}")  # Debug
        
        try:
            agent_type = AgentType(result.content.strip().lower())
            state["agent_type"] = agent_type
        except ValueError:
            state["agent_type"] = AgentType.GENERAL
            
        return state

    async def generate_agent_response(self, state: ChatState) -> AsyncGenerator[ChatState, None]:
        print("Generating agent response")
        messages = state["messages"]
        last_message = cast(HumanMessage, messages[-1])
        agent_type = state["agent_type"]
        
        prompt = self.agent_prompts[agent_type]
        chain = prompt | self.agent_llm
        
        agent_payload = {
            "message": last_message.content,
            "messages": state["history"]
        }
        
        new_state = ChatState(
            messages=state["messages"].copy(),
            agent_type=state["agent_type"],
            history=state["history"]
        )
        
        async for chunk in chain.astream(agent_payload):
            if chunk.content:
                # print(f"Agent generating chunk: {chunk.content}")
                # Yield each chunk immediately
                new_state["messages"] = messages + [AIMessage(content=chunk.content)]
                yield new_state

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(ChatState)
        
        workflow.add_node("route", self.route_message)
        workflow.add_node("generate", self.generate_agent_response)
        
        workflow.add_edge(START, "route")
        workflow.add_edge("route", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    async def generate_response(self, user_id: str, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            print(f"Starting response generation for message: {message}")
            
            # Initialize or get existing history
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            
            # Add new message to history
            self.conversation_history[user_id].append(HumanMessage(content=message))
            
            state = ChatState(
                messages=[HumanMessage(content=message)],
                agent_type="",
                history=self.conversation_history[user_id]
            )
            
            first = True
            async for msg, metadata in self.workflow.astream(state, stream_mode="messages"):
                if msg.content and not isinstance(msg, HumanMessage):
                    if msg.content not in [AgentType.CAREER, AgentType.GENERAL]:
                        yield {"content": msg.content}
                    # print(f"Yielding content chunk: {msg.content}")

                if isinstance(msg, AIMessageChunk):
                    if first:
                        gathered = msg
                        first = False
                    else:
                        gathered = gathered + msg

                    # Handle any tool calls if present
                    if msg.tool_call_chunks:
                        print(f"Tool calls: {gathered.tool_calls}")

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))