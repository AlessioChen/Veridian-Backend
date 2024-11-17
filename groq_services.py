import os
from groq import Groq
from dotenv import load_dotenv
import json
from models.user_profile import UserProfile
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

    def generate_job_suggestions(self, user_profile: UserProfile):
        job_market_data_path = f"{os.getcwd()}/datasets/yr-earnings-occupation.json"
        job_market_data = self.load_job_market_data(job_market_data_path)
        user_prompt = f"""
                    User career profile data: {user_profile}.
                    Job market data {job_market_data}.
                    """
        system_prompt = """
        You are a career advisor assistant. You will be given two types of information:
        1. Personal Career Profile:
        {
            "jobs": [
                {
                    "title": "Job title",
                    "location": "City, Region, Country",
                    "dates": {
                        "start": "MMM YYYY",
                        "end": "MMM YYYY or Present"
                    },
                    "details": [
                        "Detailed achievement or responsibility 1",
                        "Detailed achievement or responsibility 2"
                    ]
                }
            ],
            "education": {
                "level": "education level",
                "details": "specific grades or qualifications"
            },
            "skills": ["Current skills list"],
            "wanted_skills": "Desired skills list",
            "location": "current location"
        }

        2. Job Market Data:
        [
            {
                "description": "Job title/description",
                "code": "Job classification code",
                "median": Median salary as float
            }
        ]

        Your task is to:

        1. Analyze the person's career trajectory by:
           - Identifying progression patterns in their roles
           - Calculating total years of experience
           - Extracting quantifiable achievements from job details
           - Mapping skill development across roles
           - Noting industry transitions and location patterns

        2. Extract and categorize skills from their work history:
           - Technical skills mentioned in job details
           - Management and leadership capabilities
           - Quantitative achievements (e.g., "increased efficiency by X%")
           - Soft skills demonstrated through responsibilities

        3. Compare their current role against the provided job market data:
           - Identify roles with higher median salaries
           - Find positions that build on their demonstrated achievements
           - Consider location compatibility
           - Factor in educational requirements vs. their background

        4. Provide a ranked list of 1-3 recommended jobs, including:
           - Job title and median salary
           - Alignment with their proven achievements
           - How their quantifiable results transfer to the new role
           - Required skill gaps vs. their wanted skills
           - Geographic considerations based on their location history

        5. Create a detailed transition plan for each role:
           - Specific qualifications needed given their education level
           - Training programs that account for their background
           - Timeline based on their skill acquisition history
           - Local opportunities for practical experience

        Format your response as:

        Career Analysis:
        Experience: [X] years total
        Key Achievements:
        - [Quantified achievement 1]
        - [Quantified achievement 2]
        Progression Pattern: [Analysis of career progression]

        Top Recommendations:

        1. [Job Title] - £[Median Salary]
           Match Score: [X/10]
           Why This Fits:
           - [Reference specific achievement from their history]
           - [How it builds on demonstrated capabilities]
           - [Location considerations]

           Relevant Achievements:
           - [Past achievement that directly transfers]
           - [Quantified result that applies]

           Development Needs:
           - [Required qualification vs. education level]
           - [Skill gap vs. wanted skills]

           Transition Plan:
           - [Immediate step based on background]
           - [Training aligned with education level]
           - [Local opportunity to gain experience]

        [Repeat for each recommendation]

        Remember to:
        - Focus heavily on quantified achievements from job details
        - Consider geographical progression in career history
        - Account for education level in qualification requirements
        - Map progression between similar industries
        - Identity transferable skills from detailed job descriptions
        - Factor in tenure length in each role
        - Consider proximity of recommended roles to current location
        - Balance formal education with practical experience
        - Align recommendations with demonstrated progression rate
        """

        completion = self.client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.3,
            max_tokens=3500,
            top_p=0.95,
            stream=False,
            stop=None
        )

        # Collect the streaming response into a single string
        return completion.choices[0].message.content


    @staticmethod
    def load_job_market_data(file_path):
        try:
            with open(file_path, 'r') as file:
                job_data = json.load(file)
            return job_data

        except FileNotFoundError:
            raise FileNotFoundError(f"Job market data file not found at: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")