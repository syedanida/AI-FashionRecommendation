import requests
import json

# Your API key and Custom Search Engine ID
API_KEY = "AIzaSyCcSn82SDRGy-3dHncyL7S-DSTCLoToFeA"
CSE_ID = "0519af96f83544ad8"
EBAY_APP_ID = "SyedaNid-fashionR-SBX-e74d285b5-8aec1e02"

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