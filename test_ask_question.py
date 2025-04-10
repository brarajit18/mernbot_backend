import requests

API_URL = "http://localhost:5000/ask_question"
#QUESTION = "What are windows functions?"
#QUESTION = "What are recursive queries and can I write one with CTE?"
QUESTION = "What is semantic chunking?"

payload = {"query": QUESTION}
response = requests.post(API_URL, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())
