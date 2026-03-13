import requests

# Test health
print("Testing /health endpoint...")
r = requests.get('http://localhost:8080/health')
print(f"Status: {r.status_code}")
print(r.json())
print()

# Test API endpoint
print("Testing /api/verify/text endpoint...")
r = requests.get('http://localhost:8080/api/verify/text?text=Breaking%20news!')
print(f"Status: {r.status_code}")
print(r.json())
print()

# Test frontend
print("Testing frontend at / ...")
r = requests.get('http://localhost:8080/')
print(f"Status: {r.status_code}")
content_type = r.headers.get('content-type', 'unknown')
print(f"Content Type: {content_type}")
print(f"HTML Length: {len(r.text)} characters")
print(f"Frontend contains '<html': {'<html' in r.text.lower()}")
