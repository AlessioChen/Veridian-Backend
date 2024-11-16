from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional


class Date(BaseModel):
    start: str = Field()
    end: str = Field()


class Job(BaseModel):
    """
    Represents a job entry in a user's profile
    """
    title: str = Field()
    location: str = Field()
    dates: Date = Field()
    details: List[str] = Field()


class Education(BaseModel):
    """
    Represents a job entry in a user's profile
    """
    level: str = Field()
    details: str = Field()


class UserProfile(BaseModel):
    """
    Represents a user's professional profile
    """
    jobs: List[Job] = Field()
    education: Education = Field()
    skills: List[str] = Field()
    location: str = Field()
    wanted_skills: List[str] = Field()

