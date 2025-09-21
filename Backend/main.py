# Install dependencies:
# pip install fastapi uvicorn google-genai pydantic

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from google import genai
from google.genai import types

# Define the input schema
class StudentProfile(BaseModel):
    student_name: str
    skills: Optional[list[str]] = []
    education: Optional[str] = "Not Provided"
    interests: Optional[list[str]] = []
    career_goals: Optional[str] = "Not Provided"

# Initialize FastAPI app
app = FastAPI(title="AI Career Advisor API")

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://127.0.0.1:5500",  # your frontend
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
client = genai.Client(api_key="AIzaSyApTBe6jh3G3sne4WI6yabuDDjWBOZIeZo")
model = "gemini-2.5-flash-lite"

@app.post("/career-advice")
async def career_advice(profile: StudentProfile):
    try:
        # Prepare user input
        input_text = f"""
        Student Name: {profile.student_name}
        Skills: {', '.join(profile.skills) if profile.skills else 'Not Provided'}
        Education: {profile.education}
        Interests: {', '.join(profile.interests) if profile.interests else 'Not Provided'}
        Career Goals: {profile.career_goals}
        """

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=input_text)],
            ),
        ]

        tools = [
            types.Tool(googleSearch=types.GoogleSearch())
        ]

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            tools=tools,
            system_instruction=[
                types.Part.from_text(text="""You are an AI Career Advisor. 
Your task is to analyze a student’s profile (skills, education, interests, career goals) 
and return structured JSON in the following format:

{
  "student_name": "<string>",
  "overall_career_fit_score": <0-100>,
  "recommended_careers": [
    {
      "career_name": "<string>",
      "score": <0-100>,
      "reasoning": "<string>",
      "required_skills": ["<string>", "<string>"],
      "suggested_learning_paths": ["<string>", "<string>"]
    }
  ],
  "skill_gap_analysis": {
    "current_skills": ["<string>", "<string>"],
    "missing_skills": ["<string>", "<string>"],
    "recommended_courses": ["<string>", "<string>"]
  },
  "resume_feedback": {
    "score": <0-100>,
    "strengths": ["<string>", "<string>"],
    "weaknesses": ["<string>", "<string>"],
    "improvements": ["<string>", "<string>"]
  },
  "interview_preparation": {
    "practice_questions": ["<string>", "<string>"],
    "tips": ["<string>", "<string>"]
  },
  "final_recommendation": "<string>"
}

Rules:
- Always return valid JSON with no extra commentary.
- If data is missing, infer logically but mark as "Not Provided".
- Keep career suggestions practical and achievable.
- Resume feedback should highlight 2–3 key points.
- Interview preparation must be role-specific if career goal is provided."""),
            ],
        )

        # Generate response from Gemini
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text

        return {"career_advice": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the app with:
# uvicorn filename:app --reload