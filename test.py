import requests

url = 'http://127.0.0.1:5000/ask'
headers = {'Content-Type': 'application/json'}
data = {'question': 'What is ExpressJS?'}

#What are pure functions in dom?

response = requests.post(url, headers=headers, json=data)
print("response ________>",response.json())