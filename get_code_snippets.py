import requests
import json
import time
from bs4 import BeautifulSoup
import urllib.parse

# Define parameters for the API call
API_URL = "https://api.stackexchange.com/2.3/questions"

API_KEY = ""  # Replace with your actual API key

params = {
    "key": API_KEY,  # Ensure key is always included
    "order": "desc",
    "sort": "creation",
    "site": "stackoverflow",
    "pagesize": 100,  # Fetch 100 questions per page
    "page": 1  # Start from page 1
}

TAGS = ["c", "c%2B%2B", "java", "python", "javascript", "csharp"]  # Fixed encoding for "c++" and "c#"

def get_questions(max_pages=5):
    """Fetch multiple pages of questions from Stack Overflow."""
    all_questions = []
    for tag in TAGS:
        page = 1
        while page <= max_pages:
            params["tagged"] = tag
            params["page"] = page  # Update page number
            response = requests.get(API_URL, params=params)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Quota Remaining: {data.get('quota_remaining', 'Unknown')} for tag {tag}, page {page}")
                
                if data.get("quota_remaining", 1) == 0:
                    print("⚠️ API quota exhausted! Stopping requests.")
                    return all_questions
                
                questions = data.get("items", [])
                if not questions:
                    break  # Stop if no more questions
                
                all_questions.extend(questions)
                page += 1  # Move to next page
                time.sleep(1)  # Avoid hitting rate limits
            elif response.status_code == 429:
                print(f"⚠️ Rate limit hit for tag '{tag}'. Retrying after waiting...")
                time.sleep(10)
            else:
                print(f"Error fetching questions for tag {tag}: {response.status_code}")
                break
    
    print(f"Total Questions Fetched: {len(all_questions)}")
    return all_questions

def extract_code_snippets(answer_body):
    """Extracts code snippets from an answer body, filtering inline code."""
    soup = BeautifulSoup(answer_body, "html.parser")
    
    # Extract all <code> blocks
    code_blocks = soup.find_all("code")
    
    snippets = []
    for code in code_blocks:
        # Ignore inline <code> (inside <p>, <a>, etc.)
        if code.parent.name in ["pre", "div"]:  
            snippet = code.get_text().strip()
            if snippet:
                snippets.append(snippet)
    
    print("Extracted Snippets:", snippets)  # Debugging
    return snippets

def get_code_snippets():
    """Fetch questions, extract multiple answers, and collect code snippets."""
    questions = get_questions()
    snippets = []
    
    for question in questions:
        answer_ids = []
        if "accepted_answer_id" in question:
            answer_ids.append(question["accepted_answer_id"])  # Always add accepted answer first
        
        # Fetch top 2 highest voted answers (if available)
        answers_url = f"https://api.stackexchange.com/2.3/questions/{question['question_id']}/answers?order=desc&sort=votes&site=stackoverflow&filter=withbody&key={API_KEY}"
        response = requests.get(answers_url)
        
        if response.status_code == 200:
            answer_data = response.json()
            for ans in answer_data.get("items", [])[:2]:  # Get top 2 answers
                if "body" in ans:
                    code_snippets = extract_code_snippets(ans["body"])
                    snippets.extend(code_snippets)
        
        time.sleep(1)  # Avoid hitting API rate limits
    
    return snippets

# Run the scraper
snippets = get_code_snippets()

# Save to JSON file
with open("data/stack_overflow_code_snippets.json", "w", encoding="utf-8") as f:
    json.dump(snippets, f, indent=4)

print(f"Scraped {len(snippets)} code snippets from Stack Overflow.")