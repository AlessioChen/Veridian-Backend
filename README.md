Prerequisites:

Create a Groq API key and Perplexity API key and add it to the .env file.

# FastAPI Application Documentation

## Overview

This FastAPI application provides various endpoints for chat functionality, user profile management, URL searching, and audio transcript processing. The application includes CORS middleware configuration and integrates with several services including LLM (Language Model) services, Groq services, and Perplexity search services.

## Configuration

- Maximum File Size: 25MB
- CORS Origins: Configured for localhost development environments
  - http://localhost:3000
  - http://127.0.0.1:8000
  - http://localhost:8000
  - http://127.0.0.1:5500
  - http://localhost:5500

## API Endpoints

### 1. URL Search

```http
POST /url-search
```

Retrieves URLs from Perplexity in a JSON format.

**Request Body:**

```json
{
  "query": "string"
}
```

**Response:**

- 200: Successful search
  ```json
  {
    "response": "search_results"
  }
  ```
- 500: Search failure
  ```json
  {
    "error": "Failed to perform search"
  }
  ```

### 2. Chat

```http
POST /chat
```

General chat endpoint with streaming capabilities and multiagent system support.

**Request Body:**

```json
{
  "message": "string"
}
```

**Response:**

- 200: Successful chat response
  ```json
  {
    "response": "generated_response"
  }
  ```
- Error: Returns error message string

### 3. User Profile

```http
POST /user-profile
```

Creates user profile and generates job suggestions.

**Request Body:**

- UserProfile object (schema defined in models.user_profile)

**Response:**

- 200: Successful profile creation
  ```json
  {
    "suggestions": "job_suggestions"
  }
  ```
- 500: Profile creation failure
  ```json
  {
    "error": "Failed to generate job suggestions"
  }
  ```

### 4. Audio Transcript

```http
POST /transcript/
```

Uploads and processes audio files for transcription.

**Request Body:**

- File upload (multipart/form-data)
- Maximum file size: 25MB

**Response:**

- 200: Successful transcription
  ```json
  {
    "transcription": "transcribed_text"
  }
  ```
- 400: No file uploaded
  ```json
  {
    "error": "No file uploaded"
  }
  ```
- 500: Processing error
  ```json
  {
    "error": "An error occurred: error_message"
  }
  ```

### 5. Grounding Search

```http
POST /grounding-search
```

Performs a generic search using Perplexity.

**Request Body:**

```json
{
  "query": "string"
}
```

**Response:**

- 200: Successful search
  ```json
  {
    "response": "search_results"
  }
  ```
- 500: Search failure
  ```json
  {
    "error": "Failed to perform search"
  }
  ```

## Dependencies

- FastAPI
- Pydantic
- Custom Services:
  - LLMService
  - GroqServices
  - PerplexityService
  - PerplexityGenericSearch

## File Handling

- Audio files are temporarily stored in an 'uploaded_audio' directory
- Files are automatically removed after processing
- File size limit enforced: 25MB

## Error Handling

- All endpoints include try-catch blocks for error handling
- Errors are logged and returned with appropriate HTTP status codes
- Detailed error messages are provided in the response body

## Security

- CORS middleware configured for specific origins
- File size limitations implemented
- Proper error handling and input validation

## AGENTS

classDiagram
class AgentType {
<<enumeration>>
CAREER
GENERAL
RESUME
INTERVIEW
SKILLS
NETWORKING
JOB_SEARCH
}

    class ChatRequest {
        +str message
    }

    class ChatState {
        +list messages
        +str agent_type
        +list history
    }

    class LLMService {
        -ChatGroq router_llm
        -ChatGroq agent_llm
        -ChatPromptTemplate router_prompt
        -dict agent_prompts
        +__init__()
    }

    class ChatGroq {
        +str groq_api_key
        +str model_name
        +float temperature
        +int max_tokens
        +bool streaming
    }

    class ChatPromptTemplate {
        +list messages
        +from_messages()
    }

    LLMService --> ChatGroq : uses >
    LLMService --> ChatPromptTemplate : uses >
    LLMService --> AgentType : references >
    ChatState --> AgentType : uses >

    note for ChatGroq "Router uses llama-3.1-8b-instant\nAgents use llama-3.2-90b-text-preview"

    note for LLMService "Manages routing and specialized agent responses\nHandles different prompt templates per agent type"

    note for AgentType "Defines different specializations\nfor handling user requests"

    %% Relationships between prompts and agent types
    class AgentPrompts {
        <<interface>>
        +CAREER: career advisor
        +GENERAL: general assistant
        +RESUME: resume writer
        +INTERVIEW: interview expert
        +SKILLS: skill advisor
        +NETWORKING: networking expert
        +JOB_SEARCH: job strategist
    }

    LLMService --> AgentPrompts : contains >
