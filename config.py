import requests
import json

# Your API key and Custom Search Engine ID
API_KEY = "AA"
CSE_ID = "BB"
EBAY_APP_ID = "CC"

def search_products(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CSE_ID}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        results = []
        
        # Extract product links
        for item in data.get("items", []):
            title = item.get("title")
            link = item.get("link")
            results.append({"title": title, "link": link})
        
        return results
    else:
        print("Error:", response.status_code, response.text)
        return []
