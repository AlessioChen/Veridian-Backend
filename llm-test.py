import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_potential_transitions(current_salary, occupations, threshold=1.2):
    potential_jobs = []
    
    for occupation in occupations:
        if occupation['median'] > current_salary * threshold:
            salary_increase = occupation['median'] - current_salary
            increase_percentage = (salary_increase / current_salary) * 100
            potential_jobs.append({
                'title': occupation['description'],
                'salary': occupation['median'],
                'increase': salary_increase,
                'percentage': increase_percentage
            })
    
    return sorted(potential_jobs, key=lambda x: x['salary'], reverse=True)

def get_career_advice(current_job, current_salary, potential_jobs):
    client = Groq(api_key=os.getenv('GROQ_API'))
    
    prompt = f"""Given the following information:
Current job: {current_job}
Current salary: £{current_salary:,.2f}

Potential career transitions:
{json.dumps(potential_jobs[:5], indent=2)}

Please provide specific advice about:
1. Which of these transitions would be most feasible based on likely skill overlap
2. What specific skills or qualifications might be needed for each transition
3. A suggested action plan for making the transition

Please be concise but specific in your recommendations."""

    completion = client.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        messages=[
            {"role": "system", "content": "You are a career advisor specializing in helping people make strategic career transitions to increase their earnings."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )

    print("\nAnalyzing career options...\n")
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")

def main():
    load_dotenv()
    
    # Load occupation data
    occupations_data = load_json_data('datasets/json/yr-earnings-occupation.json')
    
    # Get user input
    print("Career Transition Advisor")
    print("------------------------")
    current_job = input("Enter your current job title: ")
    current_salary = float(input("Enter your current annual salary: "))
    
    # Get potential transitions directly using salary
    potential_jobs = get_potential_transitions(current_salary, occupations_data['occupations'])
    
    if not potential_jobs:
        print("\nNo potential transitions found that meet the salary increase criteria.")
        return
    
    # Get and display AI advice
    get_career_advice(current_job, current_salary, potential_jobs)

if __name__ == "__main__":
    main()
