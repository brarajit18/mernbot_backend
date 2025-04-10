import requests

API_URL = "http://localhost:5000/embed"
PDF_PATH = "RAG Chunking Strategies.pdf"  # Replace with your file

with open(PDF_PATH, "rb") as f:
    files = {"file": f}
    response = requests.post(API_URL, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())
